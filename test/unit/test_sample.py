import numpy as np
from core.sample import Sample


class TestSample:
    def test_get_img(self):
        sample = Sample('myid', np.ones((10, 20, 3)))
        img = sample.get_img(size=(100, 300))
        assert img.shape == (3, 100, 300)
        img = sample.get_img(img_format='channel_last')
        assert img.shape == (10, 20, 3)
        img = sample.get_img()
        assert img.shape == (3, 10, 20)

    def test_get(self):
        pass

    def test_normalize_bboxes(self):
        pass

    def test_visualize(self):
        pass
