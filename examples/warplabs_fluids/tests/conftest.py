import pytest
import warp as wp


@pytest.fixture(scope="session", autouse=True)
def warp_init():
    """Initialize Warp once per test session. Uses CPU if CUDA is unavailable."""
    wp.init()
    yield
