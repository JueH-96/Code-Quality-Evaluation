from collections import deque

class TreapNode:
    def __init__(self, key, priority):
        self.key = key
        self.priority = priority
        self.size = 1
        self.left = None
        self.right = None

class Treap:
    def __init__(self):
        self.root = None

    def insert(self, key, priority):
        self.root = self._insert(self.root, key, priority)

    def _insert(self, node, key, priority):
        if node is None:
            return TreapNode(key, priority)
        elif priority < node.priority:
            if key < node.key:
                node.left = self._insert(node.left, key, priority)
            else:
                node.right = self._insert(node.right, key, priority)
            node.size = 1 + self._size(node.left) + self._size(node.right)
            return node
        else:
            if key < node.key:
                left_size = self._size(node.left)
                new_node = self._insert(None, node.key, node.priority)
                new_node.left = self._insert(None, key, priority)
                new_node.right = node.right
                new_node.size = left_size + 2 + self._size(node.right)
                return new_node
            else:
                right_size = self._size(node.right)
                new_node = self._insert(None, node.key, node.priority)
                new_node.right = self._insert(None, key, priority)
                new_node.left = node.left
                new_node.size = right_size + 2 + self._size(node.left)
                return new_node

    def _size(self, node):
        return node.size if node else 0

    def count_less(self, key):
        return self._count_less(self.root, key)

    def _count_less(self, node, key):
        if node is None:
            return 0
        elif key < node.key:
            return self._count_less(node.left, key)
        else:
            return self._size(node.left) + 1 + self._count_less(node.right, key)

N = int(input())
H = list(map(int, input().split()))

treap = Treap()
priority_queue = deque()
for i in range(N):
    priority_queue.append((H[i], i))

answer = [0] * N
for i in range(N):
    key, index = priority_queue.pop()
    treap.insert(index, i)
    answer[index] = treap.count_less(index)

print(' '.join(map(str, answer)))