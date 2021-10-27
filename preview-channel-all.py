import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy.stats
import sklearn.mixture
import sys
import threadpoolctl
import tifffile
import zarr


threadpoolctl.threadpool_limits(1)

def fit_model(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(img.reshape((-1,1)))
    return gmm


def compute_threshold(gmm, imin, imax):
    means = gmm.means_[:, 0]
    covars = gmm.covariances_[:, 0, 0]
    _, i1, i2 = np.argsort(means)
    vmin, vmax = means[[i1, i2]] + covars[[i1, i2]] ** 0.5 * 2
    if vmin >= means[i2]:
        vmin = means[i2] + covars[i2] ** 0.5 * -1
    vmin = max(np.exp(vmin), imin, 0)
    vmax = min(np.exp(vmax), imax)
    return vmin, vmax


if __name__ == '__main__':

    channel = int(sys.argv[1])

    in_path = pathlib.Path(__file__).parent / "raw_images"

    print("Loading images...")
    image_paths = sorted(pathlib.Path('raw_images/').iterdir())
    images_full = [tifffile.imread(p, key=channel) for p in image_paths]
    ni = len(images_full)

    yi, xi = np.floor(np.linspace(0, images_full[0].shape, 50, endpoint=False)).astype(int).T
    images = np.array([img[yi][:, xi] for img in images_full])
    del images_full
    images_log = [np.log(img[img > 0]) for img in images]
    combined_log = np.hstack(images_log)

    print(f"Computing limits...")
    gmm = fit_model(combined_log)
    vmin, vmax = compute_threshold(gmm, np.min(images), np.max(images))
    print(f"  Channel {channel}: {vmin} - {vmax}")

    fig1, axes1 = plt.subplots(6, 7, figsize=(16, 10))
    for path, img_log in zip(image_paths, images_log):
        row, col = map(int, path.stem.split("_"))
        ax = axes1[row - 1, col - 1]
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.hist(img_log, bins=np.linspace(0, np.log(65535), 200), density=True, color='silver', histtype='stepfilled')
        ci = np.argsort(gmm.means_.squeeze())[-2:]
        x = np.linspace(*ax.get_xlim(), 200)
        order = np.argsort(gmm.means_.squeeze())
        for idx in order:
            mean = gmm.means_[idx, 0]
            var = gmm.covariances_[idx, 0, 0]
            weight = gmm.weights_[idx]
            dist = scipy.stats.norm(mean, var ** 0.5)
            y = dist.pdf(x) * weight
            ax.plot(x, y, lw=2, alpha=0.8)
        for v in vmin, vmax:
            ax.axvline(np.log(v), c='tab:green', ls=':')
        ax.plot(x, np.exp(gmm.score_samples(x.reshape((-1,1)))), color="black", ls="--")
        ax.set_xlim(0, np.log(65535))
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    ax2.hist(combined_log, bins=np.linspace(0, np.log(65535), 200), density=True, color='silver', histtype='stepfilled')
    ci = np.argsort(gmm.means_.squeeze())[-2:]
    x = np.linspace(*ax2.get_xlim(), 200)
    order = np.argsort(gmm.means_.squeeze())
    for idx in order:
        mean = gmm.means_[idx, 0]
        var = gmm.covariances_[idx, 0, 0]
        weight = gmm.weights_[idx]
        dist = scipy.stats.norm(mean, var ** 0.5)
        y = dist.pdf(x) * weight
        ax2.plot(x, y, lw=2, alpha=0.8)
    for v in vmin, vmax:
        ax2.axvline(np.log(v), c='tab:green', ls=':')
    ax2.plot(x, np.exp(gmm.score_samples(x.reshape((-1,1)))), color="black", ls="--")
    ax2.set_xlim(0, np.log(65535))

    plt.show()
