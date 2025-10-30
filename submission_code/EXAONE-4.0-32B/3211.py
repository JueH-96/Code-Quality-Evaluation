import sys
sys.setrecursionlimit(4000000)

MAX_RANGE = 20000000000

class SegNode:
    __slots__ = ['l', 'r', 'best', 'left_child', 'right_child']
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.best = (0, 0)
        self.left_child = None
        self.right_child = None

class SegmentTree:
    def __init__(self, MAX_RANGE):
        self.MAX_RANGE = MAX_RANGE
        self.root = SegNode(0, MAX_RANGE)
    
    def merge(self, a, b):
        if a[0] > b[0]:
            return a
        elif a[0] < b[0]:
            return b
        else:
            if a[1] > b[1]:
                return a
            else:
                return b

    def update(self, node, index, val):
        if node.l == node.r:
            node.best = self.merge(node.best, val)
            return
        
        mid = (node.l + node.r) // 2
        if index <= mid:
            if node.left_child is None:
                node.left_child = SegNode(node.l, mid)
            self.update(node.left_child, index, val)
        else:
            if node.right_child is None:
                node.right_child = SegNode(mid+1, node.r)
            self.update(node.right_child, index, val)
        
        left_best = node.left_child.best if node.left_child else (0, 0)
        right_best = node.right_child.best if node.right_child else (0, 0)
        node.best = self.merge(left_best, right_best)
    
    def query(self, node, ql, qr):
        if node is None or qr < node.l or ql > node.r:
            return (0, 0)
        if ql <= node.l and node.r <= qr:
            return node.best
        
        mid = (node.l + node.r) // 2
        left_res = (0, 0)
        right_res = (0, 0)
        if node.left_child and ql <= mid:
            left_res = self.query(node.left_child, ql, qr)
        if node.right_child and qr > mid:
            right_res = self.query(node.right_child, ql, qr)
        return self.merge(left_res, right_res)

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        prefix = [0] * (n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        seg_tree = SegmentTree(MAX_RANGE)
        
        dp = [0] * n
        last_arr = [0] * n
        
        dp[0] = 1
        last_arr[0] = nums[0]
        key1 = prefix[1] + last_arr[0]
        seg_tree.update(seg_tree.root, key1, (1, prefix[1]))
        
        max_ans = 1
        
        for i in range(1, n):
            T = prefix[i+1]
            candidate_tuple = seg_tree.query(seg_tree.root, 0, T)
            if candidate_tuple[0] > 0:
                dp_i = candidate_tuple[0] + 1
                last_i = T - candidate_tuple[1]
            else:
                dp_i = 1
                last_i = T
                
            dp[i] = dp_i
            last_arr[i] = last_i
            if dp_i > max_ans:
                max_ans = dp_i
                
            key_next = prefix[i+1] + last_i
            seg_tree.update(seg_tree.root, key_next, (dp_i, prefix[i+1]))
            
        return max_ans