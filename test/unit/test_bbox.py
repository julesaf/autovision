import unittest

from bbox import Bbox, Bboxes


class TestBboxes(unittest.TestCase):
    def test_get_resized(self):
        bboxes = Bboxes([
            Bbox(1, (10, 20, 15, 30), 0, 'smth'),
            Bbox(2, (10, 20, 20, 30), 0, 'smth')
        ])
        bboxes.mode = 2
        bboxes = bboxes.get_resized(0.5, 0.5)
        self.assertEqual(bboxes.coords, [(5, 10, 3, 5), (5, 10, 5, 5)])


class TestBbox(unittest.TestCase):
    def test_init(self):
        bbox = Bbox(1, (10, 20, 15, 30), 0, 'smth')
        self.assertEqual(bbox._area, 50)
        with self.assertRaises(AssertionError):
            Bbox(1, (10, 20, 15, 30, 20), 0, 'smth')

    def test_get_resized(self):
        # In mode 1
        bbox = Bbox(1, (10, 20, 15, 30), 0, 'smth')
        bbox = bbox.get_resized(0.5, 0.5)
        self.assertEqual(bbox._coord, (5, 10, 8, 15))
        self.assertEqual(bbox._area, 15)
        # In mode 2
        bbox = Bbox(1, (10, 20, 5, 10), 0, 'smth')
        bbox._mode = 2
        bbox = bbox.get_resized(0.5, 0.5)
        self.assertEqual(bbox._coord, (5, 10, 3, 5))
        self.assertEqual(bbox._area, 15)

    def test_change_to_mode(self):
        bbox = Bbox(1, (10, 20, 15, 30), 0, 'smth')
        bbox.mode = 2
        self.assertEqual(bbox._coord, (10, 20, 5, 10))
        self.assertEqual(bbox._mode, 2)
        bbox.mode = 1
        self.assertEqual(bbox._coord, (10, 20, 15, 30))
        self.assertEqual(bbox._mode, 1)


if __name__ == '__main__':
    unittest.main()
