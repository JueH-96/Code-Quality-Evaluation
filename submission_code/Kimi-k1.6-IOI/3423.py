class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        q = len(queries)
        
        # Each node in the segment tree stores four values:
        # best, best_without_left, best_without_right, best_without_both
        class SegmentTreeNode:
            def __init__(self, l, r):
                self.l = l
                self.r = r
                self.best = 0
                self.best_without_left = 0
                self.best_without_right = 0
                self.best_without_both = 0
                self.left = None
                self.right = None
        
        def merge(a, b):
            # Merge two nodes a (left) and b (right) into a new node
            best = max(a.best, b.best, a.best_without_right + b.best_without_left)
            best_without_left = max(b.best_without_left, b.best_without_right, a.best_without_left + b.best_without_both, a.best_without_both + b.best_without_both)
            best_without_right = max(a.best_without_right, a.best_without_left, a.best_without_both + b.best_without_right, a.best_without_both + b.best_without_left)
            best_without_both = max(a.best_without_both + b.best_without_both, a.best_without_both + b.best_without_left, a.best_without_right + b.best_without_both, a.best_without_right + b.best_without_left, a.best_without_right + b.best, a.best + b.best_without_left)
            best_without_both = max(best_without_both, a.best_without_both + b.best, a.best + b.best_without_both)
            best_without_both = max(best_without_both, a.best_without_left + b.best_without_right)
            best_without_both = max(best_without_both, a.best_without_both + b.best_without_both)
            # The above might have some redundant terms, but taking max ensures correctness
            return (best, best_without_left, best_without_right, best_without_both)
        
        def build(l, r):
            node = SegmentTreeNode(l, r)
            if l == r:
                val = nums[l]
                node.best = max(val, 0)
                node.best_without_left = max(0, 0)  # excluding left (only element)
                node.best_without_right = max(0, 0)
                node.best_without_both = max(0, 0)
                # Wait, for a single element:
                # best is max(val, 0)
                # best_without_left: exclude left (only element) → sum is 0
                # best_without_right: same as above
                # best_without_both: also 0
                node.best_without_left = 0
                node.best_without_right = 0
                node.best_without_both = 0
                return node
            mid = (l + r) // 2
            node.left = build(l, mid)
            node.right = build(mid+1, r)
            node.best, node.best_without_left, node.best_without_right, node.best_without_both = merge(node.left, node.right)
            return node
        
        def update(node, idx, value):
            if node.l == node.r == idx:
                node.best = max(value, 0)
                node.best_without_left = 0
                node.best_without_right = 0
                node.best_without_both = 0
                return
            mid = (node.l + node.r) // 2
            if idx <= mid:
                update(node.left, idx, value)
            else:
                update(node.right, idx, value)
            node.best, node.best_without_left, node.best_without_right, node.best_without_both = merge(node.left, node.right)
        
        def get_max(node):
            return node.best
        
        # Initialize the segment tree
        root = build(0, n-1)
        
        total = 0
        for pos, x in queries:
            update(root, pos, x)
            current_max = get_max(root)
            total = (total + current_max) % MOD
        
        return total % MOD