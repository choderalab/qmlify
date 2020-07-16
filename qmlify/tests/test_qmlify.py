"""
Unit and regression test for the qmlify package.
"""

# Import package, test suite, and other packages as needed
import qmlify
import pytest
import sys

def test_qmlify_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qmlify" in sys.modules
