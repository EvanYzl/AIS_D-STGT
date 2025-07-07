"""Test module initialization."""


from ais_dstgt import __version__


def test_version() -> None:
    """Test that version is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_package_imports() -> None:
    """Test that package can be imported without errors."""
    import ais_dstgt  # noqa: F401

    # Basic smoke test
    assert hasattr(ais_dstgt, "__version__")
    assert hasattr(ais_dstgt, "__author__")
    assert hasattr(ais_dstgt, "__email__")
