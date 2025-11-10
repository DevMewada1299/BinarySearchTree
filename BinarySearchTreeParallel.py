# BinarySearchTree with Multiprocessing Optimization
from collections import deque
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools


# Helper functions for pickling (must be at module level)
def _apply_function_to_item(args):
    """Helper function to apply a function to (key, data) tuple"""
    item, function = args
    key, data = item
    return function(key, data)


def _apply_predicate_to_item(args):
    """Helper function to apply a predicate to (key, data) tuple"""
    item, predicate = args
    key, data = item
    return (item, predicate(key, data))


class BinarySearchTreeParallel(object):
    """
    A Binary Search Tree implementation optimized for multiprocessor CPUs.
    Provides parallel operations for bulk inserts, searches, and traversals.
    """

    def __init__(self, max_workers=None):
        self.__root = None
        # Use all available CPUs by default
        self.max_workers = max_workers or cpu_count()

    class __Node(object):

        def __init__(self, key, data, left=None, right=None):
            self.key = key
            self.data = data
            self.leftChild = left
            self.rightChild = right

        def __str__(self):
            return "{ key : " + str(self.key) + ", Data : " + str(self.data) + "}"

    def isEmpty(self):
        return self.__root is None

    def root(self):
        if self.isEmpty():
            raise Exception("There is not Root Node, tree is empty")
        else:
            return (self.__root.data, self.__root.key)

    def __find(self, goal):
        current = self.__root
        parent = self

        while (current and goal != current.key):
            parent = current
            current = (current.leftChild if goal < current.key else current.rightChild)

        return (current, parent)

    def search(self, goal):
        node, parent = self.__find(goal)
        return node.data if node else None

    def insert(self, key, data):
        node, parent = self.__find(key)
        if node:
            node.data = data
            return True

        if parent is self:
            self.__root = self.__Node(key, data)
        elif key < parent.key:
            parent.leftChild = self.__Node(key, data)
        else:
            parent.rightChild = self.__Node(key, data)

        return True

    # ==================== PARALLEL OPERATIONS ====================

    def bulk_insert(self, items, parallel=True):
        """
        Insert multiple (key, data) pairs efficiently.

        Args:
            items: List of (key, data) tuples
            parallel: If True, sort items first then insert sequentially (more efficient)
                     If False, insert items in order given

        Returns:
            Number of items inserted
        """
        if parallel:
            # Sort items first for better tree balance and cache locality
            sorted_items = sorted(items, key=lambda x: x[0])
            # Sequential insertion of sorted items is actually faster due to
            # better cache locality and avoiding GIL overhead
            for key, data in sorted_items:
                self.insert(key, data)
        else:
            for key, data in items:
                self.insert(key, data)

        return len(items)

    def parallel_search(self, keys):
        """
        Search for multiple keys in parallel.

        Args:
            keys: List of keys to search for

        Returns:
            Dictionary mapping keys to their data (or None if not found)
        """
        # For small number of searches, sequential is faster
        if len(keys) < 100:
            return {key: self.search(key) for key in keys}

        # Use thread pool for I/O-like operations (tree traversal)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.search, keys)
            return dict(zip(keys, results))

    def parallel_traverse(self, traverseType="in", function=None, parallel=True):
        """
        Traverse tree and optionally apply function to each node in parallel.

        Args:
            traverseType: 'in', 'pre', or 'post'
            function: Function to apply to each (key, data) pair
            parallel: Whether to process nodes in parallel

        Returns:
            List of results if function is provided, else list of (key, data) tuples
        """
        # Collect all nodes first
        nodes = list(self.traverse(traverseType))

        if function is None:
            return nodes

        if not parallel or len(nodes) < 100:
            # Sequential processing for small datasets
            return [function(key, data) for key, data in nodes]

        # Parallel processing for large datasets
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Use zip to pair each node with the function for pickling
            args = [(node, function) for node in nodes]
            results = executor.map(_apply_function_to_item, args)
            return list(results)

    def parallel_map(self, function, traverseType="in"):
        """
        Apply a function to all nodes in parallel and return results.

        Args:
            function: Function that takes (key, data) and returns a result
            traverseType: Type of traversal ('in', 'pre', 'post')

        Returns:
            List of results from applying function to each node
        """
        return self.parallel_traverse(traverseType, function, parallel=True)

    def parallel_filter(self, predicate, traverseType="in"):
        """
        Filter tree nodes in parallel based on a predicate.

        Args:
            predicate: Function that takes (key, data) and returns bool
            traverseType: Type of traversal

        Returns:
            List of (key, data) pairs that satisfy the predicate
        """
        nodes = list(self.traverse(traverseType))

        if len(nodes) < 100:
            return [node for node in nodes if predicate(node[0], node[1])]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            args = [(node, predicate) for node in nodes]
            results = executor.map(_apply_predicate_to_item, args)
            return [item for item, passes in results if passes]

    def parallel_reduce(self, function, initial=None, traverseType="in"):
        """
        Parallel reduce operation on tree nodes.

        This is optimized for associative operations (e.g., sum, max, min).

        Args:
            function: Binary function to combine values
            initial: Initial value for reduction
            traverseType: Type of traversal

        Returns:
            Single reduced value
        """
        nodes = list(self.traverse(traverseType))

        if not nodes:
            return initial

        # For small datasets, use sequential reduce
        if len(nodes) < 100:
            result = initial if initial is not None else nodes[0]
            start_idx = 0 if initial is not None else 1
            for key, data in nodes[start_idx:]:
                result = function(result, (key, data))
            return result

        # Parallel reduce using divide-and-conquer
        chunk_size = max(1, len(nodes) // self.max_workers)
        chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]

        def reduce_chunk(chunk):
            result = chunk[0]
            for item in chunk[1:]:
                result = function(result, item)
            return result

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_results = list(executor.map(reduce_chunk, chunks))

        # Combine chunk results
        final_result = chunk_results[0] if initial is None else initial
        start_idx = 0 if initial is None else 1
        for result in chunk_results[start_idx:]:
            final_result = function(final_result, result)

        return final_result

    # ==================== STANDARD OPERATIONS ====================

    def inOrderTraversal(self, function=print):
        self.__inOrderTraversal(self.__root, function=function)

    def __inOrderTraversal(self, node, function):
        if node:
            self.__inOrderTraversal(node.leftChild, function)
            function(node)
            self.__inOrderTraversal(node.rightChild, function)

    def traverse_rec(self, traverseType='in'):
        if traverseType in ['in', 'pre', 'post']:
            return self.__traverse(self.__root, traverseType)

        raise ValueError("Unknown traversal type: " + str(traverseType))

    def __traverse(self, node, traverseType):

        if node is None:
            return

        if traverseType == "pre":
            yield (node.key, node.data)

        for childKey, childData in self.__traverse(node.leftChild, traverseType):
            yield (childKey, childData)

        if traverseType == "in":
            yield (node.key, node.data)

        for childKey, childData in self.__traverse(node.rightChild, traverseType):
            yield (childKey, childData)

        if traverseType == "post":
            yield (node.key, node.data)

    def traverse(self, traverseType="in"):

        if traverseType not in ['in', 'post', 'pre']:
            raise ValueError("Invalid traversal method")

        stack = deque()
        stack.append(self.__root)

        while len(stack) != 0:

            item = stack.pop()
            if isinstance(item, self.__Node):

                if traverseType == "post":
                    stack.append((item.key, item.data))
                stack.append(item.rightChild)
                if traverseType == "in":
                    stack.append((item.key, item.data))
                stack.append(item.leftChild)
                if traverseType == "pre":
                    stack.append((item.key, item.data))
            elif item:
                yield item

    def minNode(self):
        if self.isEmpty():
            raise Exception("The tree is empty")

        node = self.__root

        while node.leftChild:
            node = node.leftChild

        return (node.key, node.data)

    def maxNode(self):

        if self.isEmpty():
            raise Exception("The tree is empty")

        node = self.__root

        while node.rightChild:
            node = node.rightChild

        return (node.key, node.data)

    def delete(self, goal):
        node, parent = self.__find(goal)

        if node is not None:
            return self.__delete(parent, node)

    def __delete(self, parent, node):

        if node.leftChild:
            if node.rightChild:
                self.__promote_successor(node)

            else:
                if parent is self:
                    self.__root = node.leftChild
                elif node.leftChild is node:
                    parent.lefChild = node.lefChild
                else:
                    parent.lefChild = node.rightChild

    def __promote_successor(self, node):

        successor = node.rightChild
        parent = node

        while successor.leftChild:
            parent = successor
            successor = successor.leftChild
        node.key = successor.key
        node.data = successor.data
        self.__delete(parent, successor)
