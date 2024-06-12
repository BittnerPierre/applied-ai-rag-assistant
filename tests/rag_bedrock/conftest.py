import pytest

from trulens_eval import (
    Tru
)


@pytest.fixture(scope="package")
def trulens_prepare():
    tru = Tru(database_redact_keys=True)
    tru.reset_database()
    return tru
