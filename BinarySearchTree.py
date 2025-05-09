# BinarySearchTree
from collections import deque


class BinarySearchTree(object):

    def __init__(self):
        self.__root = None

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

        if parent is self:
            self.__root = self.__Node(key, data)
        elif key < parent.key:
            parent.leftChild = self.__Node(key, data)
        else:
            parent.rightChild = self.__Node(key, data)

        return True

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