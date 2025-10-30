# YOUR CODE HERE
import sys
import threading

sys.setrecursionlimit(1 << 25)

def main():
    import sys
    N, K, Q = map(int, sys.stdin.readline().split())

    # Read updates
    updates = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    positions = set()
    values_set = set()
    values_set.add(0)  # Initial value of zero
    for x_i, y_i in updates:
        values_set.add(y_i)

    # Build list of unique values
    values_list = sorted(values_set)
    value_to_index = {v: i for i, v in enumerate(values_list)}
    index_to_value = {i: v for i, v in enumerate(values_list)}
    M = len(values_list)

    # Initialize Segment Tree
    class SegmentTreeNode:
        def __init__(self, l, r):
            self.l = l
            self.r = r
            self.left = None
            self.right = None
            self.cnt = 0  # Number of elements in this interval
            self.sum = 0  # Sum of elements in this interval

    def build(l, r):
        node = SegmentTreeNode(l, r)
        if l == r:
            pass  # Leaf node
        else:
            m = (l + r) // 2
            node.left = build(l, m)
            node.right = build(m + 1, r)
        return node

    root = build(0, M -1)

    # Helper functions
    def update(node, idx, cnt_delta, sum_delta):
        if node.l == node.r:
            node.cnt += cnt_delta
            node.sum += sum_delta
        else:
            if idx <= node.left.r:
                update(node.left, idx, cnt_delta, sum_delta)
            else:
                update(node.right, idx, cnt_delta, sum_delta)
            node.cnt = node.left.cnt + node.right.cnt
            node.sum = node.left.sum + node.right.sum

    # Since initially all values are zero, we need to set counts accordingly
    idx_zero = value_to_index[0]
    # Total number of elements with value zero is N
    update(root, idx_zero, N, N * 0)

    # Initialize A
    A = [0] * (N + 1)  # 1-based indexing

    # Implement query to find sum of K largest elements
    def query(node, K_left):
        if K_left == 0 or node.cnt == 0:
            return 0
        if node.l == node.r:
            take = min(K_left, node.cnt)
            return take * index_to_value[node.l]
        if node.right.cnt >= K_left:
            return query(node.right, K_left)
        else:
            sum_right = node.right.sum
            K_left -= node.right.cnt
            sum_left = query(node.left, K_left)
            return sum_right + sum_left

    # Process updates and output f(A) after each update
    for x_i, y_i in updates:
        old_value = A[x_i]
        idx_old = value_to_index[old_value]
        idx_new = value_to_index[y_i]
        # Update the Segment Tree
        update(root, idx_old, -1, -old_value)
        update(root, idx_new, 1, y_i)
        # Update A
        A[x_i] = y_i
        # Query for sum of K largest elements
        total = query(root, min(K, root.cnt))
        print(total)

threading.Thread(target=main).start()