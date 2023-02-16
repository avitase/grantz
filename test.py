from grantz import validate_update


def test_validate_update():
    valid = validate_update(
        x=[
            (10, 11),
            (10, 12),
            (20, 33),
            (1, 2),
            (2, 2),
            (3, 2),
            (5, 4),
            (6, 4),
            (24, 2),
            (24, 3),
        ],
        dx=[
            (0, 1),
            (0, -1),
            (-1, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (0, -1),
        ],
        world_size=(25, 35),
    )

    assert valid == [False, False, True, False, True, False, True, True, False, False]


if __name__ == "__main__":
    test_validate_update()
