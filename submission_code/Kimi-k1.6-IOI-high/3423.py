class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        class SegmentTreeNode:
            __slots__ = ['l', 'r', 'no_last', 'yes_last', 'left', 'right']
            def __init__(self, l, r):
                self.l = l
                self.r = r
                self.no_last = 0
                self.yes_last = 0
                self.left = None
                self.right = None
        
        def build(l, r):
            node = SegmentTreeNode(l, r)
            if l == r:
                node.yes_last = nums[l]
                return node
            mid = (l + r) // 2
            node.left = build(l, mid)
            node.right = build(mid + 1, r)
            node.no_last = max(node.left.no_last, node.left.yes_last) + node.right.no_last
            node.yes_last = node.left.no_last + node.right.yes_last
            return node
        
        def update(node, index, value):
            if node.l == node.r == index:
                node.yes_last = value
                node.no_last = 0
                return
            mid = (node.l + node.r) // 2
            if index <= mid:
                update(node.left, index, value)
            else:
                update(node.right, index, value)
            node.no_last = max(node.left.no_last, node.left.yes_last) + node.right.no_last
            node.yes_last = node.left.no_last + node.right.yes_last
        
        root = build(0, len(nums) - 1)
        total = 0
        for pos, x in queries:
            update(root, pos, x)
            current = max(root.no_last, root.yes_last)
            total = (total + current) % MOD
        return total % MOD