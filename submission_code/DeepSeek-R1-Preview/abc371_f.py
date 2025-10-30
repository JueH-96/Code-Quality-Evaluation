import sys
import random

class TreapNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.priority = random.randint(0, 1 << 30)
        self.left = None
        self.right = None
        self.size = 1
        self.sum = value

def update(node):
    if node is None:
        return
    node.size = 1
    node.sum = node.value
    if node.left is not None:
        node.size += node.left.size
        node.sum += node.left.sum
    if node.right is not None:
        node.size += node.right.size
        node.sum += node.right.sum

def split(node, key):
    if node is None:
        return (None, None)
    if node.key < key:
        left, right = split(node.right, key)
        node.right = left
        update(node)
        return (node, right)
    else:
        left, right = split(node.left, key)
        node.left = right
        update(node)
        return (left, node)

def merge(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        left.right = merge(left.right, right)
        update(left)
        return left
    else:
        right.left = merge(left, right.left)
        update(right)
        return right

def insert(root, key, value):
    left, right = split(root, key)
    new_node = TreapNode(key, value)
    return merge(merge(left, new_node), right)

def delete(root, key):
    left, right = split(root, key)
    left_left, left_right = split(left, key - 1)
    return merge(left_left, right)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr + N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    tasks = []
    for _ in range(Q):
        T = int(input[ptr]) - 1
        ptr += 1
        G = int(input[ptr])
        ptr += 1
        tasks.append((T, G))
    
    root = None
    for i in range(N):
        root = insert(root, i, X[i])
    
    total_movement = 0
    for T, G in tasks:
        left, temp = split(root, T)
        mid, right = split(temp, T + 1)
        if mid is None:
            old_pos = 0
        else:
            old_pos = mid.value
        movement = abs(G - old_pos)
        L = left.size if left is not None else 0
        R = right.size if right is not None else 0
        movement_left = (L * (L + 1)) // 2
        movement_right = (R * (R + 1)) // 2
        total_movement += movement + movement_left + movement_right
        new_node = TreapNode(T, G)
        root = merge(merge(left, new_node), right)
    
    print(total_movement)

if __name__ == '__main__':
    main()