#!/usr/bin/env python3
"""
Benchmark script comparing original BST vs parallel-optimized BST
"""

from BinarySearchTree import BinarySearchTree
from BinarySearchTreeParallel import BinarySearchTreeParallel
import time
import random
from multiprocessing import cpu_count


# Helper functions at module level for pickling
def expensive_function(key, data):
    """Simulate expensive computation"""
    result = data
    for _ in range(100):
        result = (result * 1.1 + 1) ** 0.5
    return result


def is_prime(key, data):
    """Check if data is prime (CPU-intensive)"""
    if data < 2:
        return False
    for i in range(2, int(data ** 0.5) + 1):
        if data % i == 0:
            return False
    return True


def sum_squares(acc, item):
    """Sum of squares (moderately expensive)"""
    key, data = item if isinstance(item, tuple) else (None, item)
    return acc + (data ** 2)


def benchmark_bulk_insert(size=10000):
    """Benchmark bulk insertion"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Bulk Insert ({size} items)")
    print(f"{'='*60}")

    items = [(i, f"data_{i}") for i in random.sample(range(size * 10), size)]

    # Original BST
    bst_original = BinarySearchTree()
    start = time.time()
    for key, data in items:
        bst_original.insert(key, data)
    time_original = time.time() - start

    # Parallel BST (sorted insertion)
    bst_parallel = BinarySearchTreeParallel()
    start = time.time()
    bst_parallel.bulk_insert(items, parallel=True)
    time_parallel = time.time() - start

    print(f"Original BST:  {time_original:.4f}s")
    print(f"Parallel BST:  {time_parallel:.4f}s (sorted insert)")
    speedup = time_original / time_parallel if time_parallel > 0 else 0
    print(f"Speedup:       {speedup:.2f}x")


def benchmark_multiple_searches(tree_size=10000, search_count=1000):
    """Benchmark searching for multiple keys"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Multiple Searches")
    print(f"Tree size: {tree_size}, Searches: {search_count}")
    print(f"{'='*60}")

    # Setup trees
    bst_original = BinarySearchTree()
    bst_parallel = BinarySearchTreeParallel()

    for i in range(tree_size):
        bst_original.insert(i, f"value_{i}")
        bst_parallel.insert(i, f"value_{i}")

    search_keys = random.sample(range(tree_size), search_count)

    # Original BST
    start = time.time()
    results_original = [bst_original.search(key) for key in search_keys]
    time_original = time.time() - start

    # Parallel BST
    start = time.time()
    results_parallel = bst_parallel.parallel_search(search_keys)
    time_parallel = time.time() - start

    print(f"Original BST:  {time_original:.4f}s")
    print(f"Parallel BST:  {time_parallel:.4f}s")
    speedup = time_original / time_parallel if time_parallel > 0 else 0
    print(f"Speedup:       {speedup:.2f}x")


def benchmark_map_operation(tree_size=5000):
    """Benchmark map operation (apply function to all nodes)"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Map Operation ({tree_size} nodes)")
    print(f"{'='*60}")

    # Setup trees
    bst_original = BinarySearchTree()
    bst_parallel = BinarySearchTreeParallel()

    for i in range(tree_size):
        bst_original.insert(i, i)
        bst_parallel.insert(i, i)

    # Original BST
    start = time.time()
    results_original = [expensive_function(k, d) for k, d in bst_original.traverse()]
    time_original = time.time() - start

    # Parallel BST
    start = time.time()
    results_parallel = bst_parallel.parallel_map(expensive_function)
    time_parallel = time.time() - start

    print(f"Original BST:  {time_original:.4f}s")
    print(f"Parallel BST:  {time_parallel:.4f}s")
    speedup = time_original / time_parallel if time_parallel > 0 else 0
    print(f"Speedup:       {speedup:.2f}x")


def benchmark_filter_operation(tree_size=10000):
    """Benchmark filter operation"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Filter Operation ({tree_size} nodes)")
    print(f"{'='*60}")

    # Setup trees
    bst_original = BinarySearchTree()
    bst_parallel = BinarySearchTreeParallel()

    for i in range(tree_size):
        bst_original.insert(i, random.randint(1, 1000))
        bst_parallel.insert(i, random.randint(1, 1000))

    # Original BST
    start = time.time()
    results_original = [(k, d) for k, d in bst_original.traverse() if is_prime(k, d)]
    time_original = time.time() - start

    # Parallel BST
    start = time.time()
    results_parallel = bst_parallel.parallel_filter(is_prime)
    time_parallel = time.time() - start

    print(f"Original BST:  {time_original:.4f}s (found {len(results_original)} primes)")
    print(f"Parallel BST:  {time_parallel:.4f}s (found {len(results_parallel)} primes)")
    speedup = time_original / time_parallel if time_parallel > 0 else 0
    print(f"Speedup:       {speedup:.2f}x")


def benchmark_reduce_operation(tree_size=10000):
    """Benchmark reduce operation"""
    print(f"\n{'='*60}")
    print(f"Benchmark: Reduce Operation ({tree_size} nodes)")
    print(f"{'='*60}")

    # Setup trees
    bst_original = BinarySearchTree()
    bst_parallel = BinarySearchTreeParallel()

    for i in range(1, tree_size + 1):
        bst_original.insert(i, i)
        bst_parallel.insert(i, i)

    # Original BST
    start = time.time()
    result_original = 0
    for k, d in bst_original.traverse():
        result_original = sum_squares(result_original, (k, d))
    time_original = time.time() - start

    # Parallel BST
    start = time.time()
    result_parallel = bst_parallel.parallel_reduce(sum_squares, initial=0)
    time_parallel = time.time() - start

    print(f"Original BST:  {time_original:.4f}s (result: {result_original})")
    print(f"Parallel BST:  {time_parallel:.4f}s (result: {result_parallel})")
    speedup = time_original / time_parallel if time_parallel > 0 else 0
    print(f"Speedup:       {speedup:.2f}x")


def run_all_benchmarks():
    """Run all benchmarks"""
    print("\n" + "="*60)
    print("BINARY SEARCH TREE - MULTIPROCESSOR OPTIMIZATION BENCHMARKS")
    print("="*60)
    print(f"System: {cpu_count()} CPU cores available")
    print("="*60)

    benchmark_bulk_insert(size=10000)
    benchmark_multiple_searches(tree_size=10000, search_count=1000)
    benchmark_map_operation(tree_size=5000)
    benchmark_filter_operation(tree_size=10000)
    benchmark_reduce_operation(tree_size=10000)

    print("\n" + "="*60)
    print("BENCHMARKS COMPLETED")
    print("="*60)
    print("\nNotes:")
    print("- Speedup depends on CPU core count and operation type")
    print("- CPU-intensive operations benefit most from parallelization")
    print("- Small datasets may show overhead from parallel processing")
    print("- Bulk insert uses sorted insertion for better cache locality")
    print("="*60)


if __name__ == "__main__":
    run_all_benchmarks()
