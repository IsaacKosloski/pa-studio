def test_characterization_sees_core() -> None:
    import characterization
    import core

    assert core.__name__ and characterization.__name__
