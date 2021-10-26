import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy.stats
import sklearn.mixture
import sys
import threadpoolctl
import tifffile
import tqdm
import zarr


threadpoolctl.threadpool_limits(1)

def fit_threshold(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(img.reshape((-1,1)))
    return gmm


if __name__ == '__main__':

    channel = int(sys.argv[1])

    in_path = pathlib.Path(__file__).parent / "raw_images"

    fig, axes = plt.subplots(6, 7, figsize=(16, 10))
    image_paths = sorted(pathlib.Path('raw_images/').iterdir())
    images = [tifffile.imread(p, key=channel) for p in image_paths]
    ni = len(images)

    yi, xi = np.floor(np.linspace(0, images[0].shape, 200, endpoint=False)).astype(int).T
    images = [img[yi][:, xi] for img in images]
    images_log = [np.log(img[img > 0]) for img in images]

    print(f"Computing limits for individual images")
    pool = concurrent.futures.ProcessPoolExecutor()
    progress = tqdm.tqdm(
        zip(pool.map(fit_threshold, images_log), image_paths, images, images_log),
        total=ni
    )
    limits = []
    for i, (gmm, path, img, img_log) in enumerate(progress):

        means = gmm.means_[:, 0]
        covars = gmm.covariances_[:, 0, 0]
        _, i1, i2 = np.argsort(means)

        vmin, vmax = means[[i1, i2]] + covars[[i1, i2]] ** 0.5 * 2
        if vmin >= means[i2]:
            vmin = means[i2] + covars[i2] ** 0.5 * -1
        vmin = max(np.exp(vmin), img.min(), 0)
        vmax = min(np.exp(vmax), img.max())
        limits.append((vmin, vmax))

        row, col = map(int, path.stem.split("_"))
        ax = axes[row - 1, col - 1]

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

    fig.tight_layout()

    fig, ax = plt.subplots()
    limits = np.array(limits)
    ax.hist(np.log(limits), bins=40, histtype="step", label=["vmin", "vmax"])
    ax.legend()

    plt.show()
