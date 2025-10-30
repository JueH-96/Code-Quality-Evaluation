from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        INF = 10**18  # A large value to represent negative infinity

        class SegmentTreeNode:
            __slots__ = ['l', 'r', 'left', 'right', 'max1', 'max2', 'max3', 'max4']
            def __init__(self, l, r):
                self.l = l
                self.r = r
                self.left = None
                self.right = None
                self.max1 = 0  # first and last taken
                self.max2 = 0  # first taken, last not
                self.max3 = 0  # first not, last taken
                self.max4 = 0  # first and last not

        def build(l, r):
            node = SegmentTreeNode(l, r)
            if l == r:
                val = nums[l]
                node.max1 = val
                node.max2 = -INF
                node.max3 = -INF
                node.max4 = 0
            else:
                mid = (l + r) // 2
                node.left = build(l, mid)
                node.right = build(mid + 1, r)
                l1, l2, l3, l4 = node.left.max1, node.left.max2, node.left.max3, node.left.max4
                r1, r2, r3, r4 = node.right.max1, node.right.max2, node.right.max3, node.right.max4
                # Compute max1
                option1 = l1 + r3
                option2 = l2 + max(r1, r3)
                node.max1 = max(option1, option2)
                # Compute max2
                option1a = l1 + r4
                option2a = l2 + max(r2, r4)
                node.max2 = max(option1a, option2a)
                # Compute max3
                option1b = l3 + r3
                option2b = l4 + max(r1, r3)
                node.max3 = max(option1b, option2b)
                # Compute max4
                option1c = l3 + r4
                option2c = l4 + max(r2, r4)
                node.max4 = max(option1c, option2c)
            return node

        def update(node, pos, val):
            if node.l == node.r == pos:
                node.max1 = val
                node.max2 = -INF
                node.max3 = -INF
                node.max4 = 0
            else:
                if pos <= node.left.r:
                    update(node.left, pos, val)
                else:
                    update(node.right, pos, val)
                l1, l2, l3, l4 = node.left.max1, node.left.max2, node.left.max3, node.left.max4
                r1, r2, r3, r4 = node.right.max1, node.right.max2, node.right.max3, node.right.max4
                # Recompute max1
                option1 = l1 + r3
                option2 = l2 + max(r1, r3)
                node.max1 = max(option1, option2)
                # Recompute max2
                option1a = l1 + r4
                option2a = l2 + max(r2, r4)
                node.max2 = max(option1a, option2a)
                # Recompute max3
                option1b = l3 + r3
                option2b = l4 + max(r1, r3)
                node.max3 = max(option1b, option2b)
                # Recompute max4
                option1c = l3 + r4
                option2c = l4 + max(r2, r4)
                node.max4 = max(option1c, option2c)

        root = build(0, n - 1)
        res = 0
        for pos, x in queries:
            update(root, pos, x)
            current_max = max(root.max1, root.max2, root.max3, root.max4)
            res = (res + current_max) % MOD
        return res