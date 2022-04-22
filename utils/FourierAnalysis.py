'''
@Description  : fft script
@Author       : Chi Liu
@Date         : 2022-04-06 17:44:13
@LastEditTime : 2022-04-06 20:12:42
'''
from mimetypes import init
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class FourierAnalysis(object):

    def __init__(self, im_arr, bandwidth, to_path) -> None:
        if isinstance(im_arr, str):
            im_arr = np.array(Image.open(im_arr))  # read from path
        else:
            assert isinstance(im_arr, np.ndarray)
        assert im_arr.shape[-1] == 3  # RGB image

        self.im_arr = im_arr
        self.bandwidth = bandwidth
        self.to_path = to_path
        self.fre = np.fft.fft2(im_arr, axes=(0, 1))
        self.fre_shift = np.fft.fftshift(self.fre)
        self.amp, self.pha = self.decompose()
        self.iamp, self.ipha = self.idecompose()
        self.mask = self.circular_mask()
        self.low_fre = self.fre_shift * self.mask
        self.high_fre = self.fre_shift * (1 - self.mask)
        self.low_img = np.abs(
            np.fft.ifft2(np.fft.ifftshift(self.low_fre), axes=(0, 1)))
        self.high_img = np.abs(
            np.fft.ifft2(np.fft.ifftshift(self.high_fre), axes=(0, 1)))
        self.plot()

    def decompose(self):
        return np.abs(self.fre), np.angle(self.fre)

    def idecompose(self):
        amp_constant = self.amp.mean()
        amp_fre = amp_constant * np.e**(1j * self.pha)
        iamp = np.abs(np.fft.ifft2(amp_fre, axes=(0, 1)))

        pha_constant = self.pha.mean()
        pha_fre = self.amp * np.e**(1j * pha_constant)
        ipha = np.abs(np.fft.ifft2(pha_fre, axes=(0, 1)))
        return iamp, ipha

    def plot(self):
        title_disk = {
            'original image': self.im_arr,
            'Fourier spectrum': self.fre_shift,
            'Amplitude-only component': self.iamp,
            'Phase-only component': self.ipha,
            'Filter': self.mask,
            'High-frequency spectrum': self.high_fre,
            'Low-frequency spectrum': self.low_fre,
            'High-frequency component': self.high_img,
            'Low_frequency component': self.low_img
        }
        fig = plt.figure(figsize=(18, 18))
        for i, t in enumerate(title_disk.keys()):
            ax = fig.add_subplot(3, 3, i + 1)
            ax.set_title(t)
            view = title_disk[t]
            if i in [1, 5, 6]:
                view = np.log(1 + np.abs(view))
                view = (view - view.min()) / (view.max() - view.min()) * 255
            if i == 4: view = view * 255
            view = np.clip(view, 0, 255)
            view = view.astype('uint8')
            ax.imshow(view)
        plt.show()
        plt.savefig(self.to_path, bbox_inches='tight')

    def circular_mask(self):
        h, w, _ = self.im_arr.shape
        Y, X = np.ogrid[:h, :w]
        center = (int(w / 2), int(h / 2))
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= self.bandwidth
        return np.dstack([mask, mask, mask])


if __name__ == '__main__':
    im = './test2.jpg'
    FA = FourierAnalysis(im, 20, './re.png')