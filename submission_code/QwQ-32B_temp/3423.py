class Node:
    __slots__ = ['l', 'r', 'left', 'right', 'taken', 'not_taken', 'max_val']
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.left = None
        self.right = None
        self.taken = 0
        self.not_taken = 0
        self.max_val = 0

class Solution:
    def maximumSumSubsequence(self, nums, queries):
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        root = self.build(nums, 0, n - 1)
        res = 0
        for pos, x in queries:
            self.update(root, pos, x)
            res = (res + root.max_val) % MOD
        return res
    
    def build(self, nums, l, r):
        node = Node(l, r)
        if l == r:
            x = nums[l]
            node.taken = max(0, x)
            node.not_taken = 0
            node.max_val = node.taken
        else:
            mid = (l + r) // 2
            node.left = self.build(nums, l, mid)
            node.right = self.build(nums, mid + 1, r)
            node.taken = node.left.not_taken + node.right.taken
            node.not_taken = max(node.left.taken + node.right.not_taken, node.left.not_taken + node.right.not_taken)
            node.max_val = max(node.taken, node.not_taken)
        return node
    
    def update(self, node, idx, new_val):
        if node.l == node.r == idx:
            x = new_val
            node.taken = max(0, x)
            node.not_taken = 0
            node.max_val = node.taken
        else:
            mid = (node.l + node.r) // 2
            if idx <= mid:
                self.update(node.left, idx, new_val)
            else:
                self.update(node.right, idx, new_val)
            # Update current node's values
            node.taken = node.left.not_taken + node.right.taken
            node.not_taken = max(node.left.taken + node.right.not_taken, node.left.not_taken + node.right.not_taken)
            node.max_val = max(node.taken, node.not_taken)