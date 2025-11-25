def test_sanity():
    """
    A simple test to ensure the testing framework is working correctly.
    """
    assert 1 + 1 == 2

def test_import_src():
    """
    Ensure we can import from the src directory.
    """
    try:
        import src
        assert True
    except ImportError:
        assert False, "Failed to import src package"
