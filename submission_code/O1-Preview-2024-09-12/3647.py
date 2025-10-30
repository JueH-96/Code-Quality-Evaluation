class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        from collections import defaultdict

        # Step 1: Initialize capacity[i] = number of queries covering i
        capacity = [0] * n
        for l, r in queries:
            capacity[l] += 1
            if r + 1 < n:
                capacity[r + 1] -= 1
        for i in range(1, n):
            capacity[i] += capacity[i - 1]

        # Step 2: Check if capacity[i] >= nums[i]
        for i in range(n):
            if capacity[i] < nums[i]:
                return -1  # Not possible to make nums zero
        # Step 3: surplus[i] = capacity[i] - nums[i]
        surplus = [capacity[i] - nums[i] for i in range(n)]

        # Step 4: Build segment tree over surplus[i]
        class SegmentTreeNode:
            def __init__(self, left, right):
                self.left = left
                self.right = right
                self.left_child = None
                self.right_child = None
                self.min_value = 0
                self.lazy = 0  # For lazy propagation

        def build(node, surplus):
            if node.left == node.right:
                node.min_value = surplus[node.left]
            else:
                mid = (node.left + node.right) // 2
                node.left_child = SegmentTreeNode(node.left, mid)
                node.right_child = SegmentTreeNode(mid + 1, node.right)
                build(node.left_child, surplus)
                build(node.right_child, surplus)
                node.min_value = min(node.left_child.min_value, node.right_child.min_value)

        def update(node, l, r, val):
            if node.left > r or node.right < l:
                return
            if node.left >= l and node.right <= r:
                node.min_value += val
                node.lazy += val
            else:
                propagate(node)
                update(node.left_child, l, r, val)
                update(node.right_child, l, r, val)
                node.min_value = min(node.left_child.min_value, node.right_child.min_value)

        def query(node, l, r):
            if node.left > r or node.right < l:
                return float('inf')
            if node.left >= l and node.right <= r:
                return node.min_value
            propagate(node)
            return min(query(node.left_child, l, r), query(node.right_child, l, r))

        def propagate(node):
            if node.lazy != 0:
                if node.left_child is not None:
                    node.left_child.min_value += node.lazy
                    node.left_child.lazy += node.lazy
                    node.right_child.min_value += node.lazy
                    node.right_child.lazy += node.lazy
                node.lazy = 0

        # Build segment tree
        root = SegmentTreeNode(0, n - 1)
        build(root, surplus)

        # Step 6: Process queries in reverse order
        answer = len(queries)
        # For inverse mapping of queries to indices in original list
        reversed_queries = queries[::-1]
        for l, r in reversed_queries:
            min_surplus = query(root, l, r)
            if min_surplus > 0:
                # We can remove this query
                update(root, l, r, -1)
                answer -= 1
            # Else we need to keep this query

        num_removed = len(queries) - answer
        return num_removed