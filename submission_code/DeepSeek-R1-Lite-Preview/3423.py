class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        N = len(nums)
        
        class SegmentTreeNode:
            def __init__(self, l, r):
                self.l = l
                self.r = r
                self.left = None
                self.right = None
                self.include = 0
                self.exclude = 0
        
        def build(l, r):
            node = SegmentTreeNode(l, r)
            if l == r:
                node.include = nums[l]
                node.exclude = 0
            else:
                mid = (l + r) // 2
                node.left = build(l, mid)
                node.right = build(mid + 1, r)
                node.exclude = max(node.left.exclude, node.left.include) + max(node.right.exclude, node.right.include)
                node.include = node.left.exclude + node.right.include
            return node
        
        def update(node, pos, value):
            if node.l == node.r:
                node.include = value
                node.exclude = 0
            else:
                mid = (node.l + node.r) // 2
                if pos <= mid:
                    update(node.left, pos, value)
                else:
                    update(node.right, pos, value)
                node.exclude = max(node.left.exclude, node.left.include) + max(node.right.exclude, node.right.include)
                node.include = node.left.exclude + node.right.include
        
        def query(node):
            return max(node.include, node.exclude)
        
        root = build(0, N - 1)
        total = 0
        for q in queries:
            pos, x = q
            nums[pos] = x
            update(root, pos, x)
            current_max = max(0, query(root))
            total = (total + current_max) % MOD
        return total