from core.annotations.bbox import Bbox


class TestBbox:
    def test_get_resized(self):
        # In box_format 1
        bbox = Bbox(1, [10, 20, 15, 30], 'smth', mode=1)
        bbox = bbox.get_resized(0.5, 0.5)
        assert bbox._coord == [5, 10, 8, 15]
        assert bbox._area == 15
        # In box_format 2
        bbox = Bbox(1, [10, 20, 5, 10], 'smth', mode=2)
        bbox = bbox.get_resized(0.5, 0.5)
        assert bbox._coord == [5, 10, 3, 5]
        assert bbox._area == 15

    def test_normalize_coord(self):
        # In box_format 1
        bbox = Bbox(1, [10, 20, 15, 30], 'smth', mode=1)
        bbox.normalize_coord(img_size=(100, 200))
        assert bbox.is_coord_normalized
        assert bbox._coord == [0.05, 0.2, 0.075, 0.3]
        # Check redo
        bbox.normalize_coord(img_size=(100, 200))
        assert bbox.is_coord_normalized
        assert bbox._coord == [0.05, 0.2, 0.075, 0.3]
        # In box_format 2
        bbox = Bbox(1, [10, 20, 5, 10], 'smth', mode=2)
        bbox.normalize_coord(img_size=(100, 200))
        assert bbox._coord == [0.05, 0.2, 0.05, 0.05]

    def test_set_mode(self):
        bbox = Bbox(1, [10, 20, 15, 30], 'smth', mode=1)
        bbox.box_format = 2
        assert bbox._coord == [10, 20, 5, 10]
        assert bbox._box_format == 2
        bbox.box_format = 1
        assert bbox._coord == [10, 20, 15, 30]
        assert bbox._box_format == 1

