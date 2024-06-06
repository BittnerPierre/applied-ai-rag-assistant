import pytest

from trulens_eval import (
    Tru
)


@pytest.fixture(scope="package")
def trulens_prepare():
    tru = Tru()
    tru.reset_database()
    return tru
