#!/usr/bin/env python3
"""
Quick test to verify parallel BST implementation works correctly
"""

from BinarySearchTreeParallel import BinarySearchTreeParallel
import random


# Helper functions at module level for pickling
def square(key, data):
    """Square the data value"""
    return data ** 2


def is_even(key, data):
    """Check if data is even"""
    return data % 2 == 0


def add(acc, item):
    """Add data to accumulator"""
    key, data = item if isinstance(item, tuple) else (None, item)
    return acc + data


def test_basic_operations():
    """Test basic BST operations"""
    print("Testing basic operations...")
    bst = BinarySearchTreeParallel()

    # Test insert and search
    bst.insert(5, "five")
    bst.insert(3, "three")
    bst.insert(7, "seven")

    assert bst.search(5) == "five"
    assert bst.search(3) == "three"
    assert bst.search(7) == "seven"
    assert bst.search(99) is None

    print("✓ Basic operations work")


def test_bulk_insert():
    """Test bulk insert"""
    print("Testing bulk insert...")
    bst = BinarySearchTreeParallel()

    items = [(i, f"data_{i}") for i in range(100)]
    bst.bulk_insert(items, parallel=True)

    # Verify all items were inserted
    nodes = list(bst.traverse())
    assert len(nodes) == 100

    print(f"✓ Bulk insert works ({len(nodes)} items)")


def test_parallel_search():
    """Test parallel search"""
    print("Testing parallel search...")
    bst = BinarySearchTreeParallel()

    # Insert test data
    for i in range(1000):
        bst.insert(i, f"value_{i}")

    # Search for multiple keys
    search_keys = [0, 100, 500, 999]
    results = bst.parallel_search(search_keys)

    assert results[0] == "value_0"
    assert results[500] == "value_500"
    assert results[999] == "value_999"

    print("✓ Parallel search works")


def test_parallel_map():
    """Test parallel map"""
    print("Testing parallel map...")
    bst = BinarySearchTreeParallel()

    for i in range(100):
        bst.insert(i, i)

    results = bst.parallel_map(square)

    assert len(results) == 100
    assert results[0] == 0
    assert results[10] == 100  # 10^2

    print("✓ Parallel map works")


def test_parallel_filter():
    """Test parallel filter"""
    print("Testing parallel filter...")
    bst = BinarySearchTreeParallel()

    for i in range(100):
        bst.insert(i, i)

    even_nodes = bst.parallel_filter(is_even)

    assert len(even_nodes) == 50
    assert all(data % 2 == 0 for key, data in even_nodes)

    print("✓ Parallel filter works")


def test_parallel_reduce():
    """Test parallel reduce"""
    print("Testing parallel reduce...")
    bst = BinarySearchTreeParallel()

    for i in range(1, 11):
        bst.insert(i, i)

    total = bst.parallel_reduce(add, initial=0)

    assert total == sum(range(1, 11))  # 1+2+...+10 = 55

    print(f"✓ Parallel reduce works (sum = {total})")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Quick Test Suite - Parallel BST")
    print("="*50 + "\n")

    try:
        test_basic_operations()
        test_bulk_insert()
        test_parallel_search()
        test_parallel_map()
        test_parallel_filter()
        test_parallel_reduce()

        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
