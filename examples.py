#!/usr/bin/env python3
"""
Examples demonstrating parallel BST operations
"""

from BinarySearchTreeParallel import BinarySearchTreeParallel
import time
import random


# Helper functions at module level for pickling
def square_value(key, data):
    """Square the value"""
    return key, data ** 2


def is_even(key, data):
    """Check if data is even"""
    return data % 2 == 0


def greater_than_500(key, data):
    """Check if data > 500"""
    return data > 500


def add(acc, item):
    """Add values"""
    key, data = item if isinstance(item, tuple) else (None, item)
    return acc + data


def example_bulk_insert():
    """Example: Bulk insert with parallel optimization"""
    print("\n=== Example 1: Bulk Insert ===")

    bst = BinarySearchTreeParallel()

    # Generate 10,000 random items
    items = [(i, f"data_{i}") for i in random.sample(range(100000), 10000)]

    start = time.time()
    bst.bulk_insert(items, parallel=True)
    elapsed = time.time() - start

    print(f"Inserted {len(items)} items in {elapsed:.4f} seconds")
    print(f"Tree size verification: {len(list(bst.traverse()))} nodes")


def example_parallel_search():
    """Example: Search for multiple keys in parallel"""
    print("\n=== Example 2: Parallel Search ===")

    bst = BinarySearchTreeParallel()

    # Insert test data
    for i in range(10000):
        bst.insert(i, f"value_{i}")

    # Search for 1000 random keys
    search_keys = random.sample(range(10000), 1000)

    start = time.time()
    results = bst.parallel_search(search_keys)
    elapsed = time.time() - start

    found = sum(1 for v in results.values() if v is not None)
    print(f"Searched {len(search_keys)} keys in {elapsed:.4f} seconds")
    print(f"Found {found} keys")


def example_parallel_map():
    """Example: Apply function to all nodes in parallel"""
    print("\n=== Example 3: Parallel Map ===")

    bst = BinarySearchTreeParallel()

    # Insert test data
    for i in range(1000):
        bst.insert(i, i * 2)

    # Square all values in parallel
    start = time.time()
    results = bst.parallel_map(square_value)
    elapsed = time.time() - start

    print(f"Applied function to {len(results)} nodes in {elapsed:.4f} seconds")
    print(f"First 10 results: {results[:10]}")


def example_parallel_filter():
    """Example: Filter nodes in parallel"""
    print("\n=== Example 4: Parallel Filter ===")

    bst = BinarySearchTreeParallel()

    # Insert test data
    for i in range(5000):
        bst.insert(i, i)

    # Find all even numbers in parallel
    start = time.time()
    even_nodes = bst.parallel_filter(is_even)
    elapsed = time.time() - start

    print(f"Filtered {len(list(bst.traverse()))} nodes in {elapsed:.4f} seconds")
    print(f"Found {len(even_nodes)} even numbers")


def example_parallel_reduce():
    """Example: Parallel reduce operation"""
    print("\n=== Example 5: Parallel Reduce ===")

    bst = BinarySearchTreeParallel()

    # Insert test data
    for i in range(1, 1001):
        bst.insert(i, i)

    # Sum all values in parallel
    start = time.time()
    total = bst.parallel_reduce(add, initial=0)
    elapsed = time.time() - start

    expected = sum(range(1, 1001))
    print(f"Reduced {len(list(bst.traverse()))} nodes in {elapsed:.4f} seconds")
    print(f"Sum: {total} (expected: {expected})")


def example_complex_operation():
    """Example: Complex operation combining multiple parallel operations"""
    print("\n=== Example 6: Complex Parallel Operation ===")

    bst = BinarySearchTreeParallel(max_workers=4)

    # Bulk insert
    print("Step 1: Bulk inserting 5000 items...")
    items = [(i, random.randint(1, 1000)) for i in range(5000)]
    bst.bulk_insert(items, parallel=True)

    # Filter for values > 500
    print("Step 2: Filtering values > 500...")
    filtered = bst.parallel_filter(greater_than_500)
    print(f"Found {len(filtered)} values > 500")

    # Calculate average of filtered values
    print("Step 3: Calculating average...")
    if filtered:
        total = sum(data for key, data in filtered)
        average = total / len(filtered)
        print(f"Average of filtered values: {average:.2f}")


if __name__ == "__main__":
    print("Binary Search Tree - Multiprocessor Optimization Examples")
    print("=" * 60)

    example_bulk_insert()
    example_parallel_search()
    example_parallel_map()
    example_parallel_filter()
    example_parallel_reduce()
    example_complex_operation()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
