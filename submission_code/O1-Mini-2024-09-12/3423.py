from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        class SegmentTreeNode:
            def __init__(self, left, right):
                self.left = left
                self.right = right
                self.left_child = None
                self.right_child = None
                self.include = 0
                self.exclude = 0
        
        def build(node, nums):
            if node.left == node.right:
                node.include = max(nums[node.left], 0)
                node.exclude = 0
                return
            mid = (node.left + node.right) // 2
            node.left_child = SegmentTreeNode(node.left, mid)
            node.right_child = SegmentTreeNode(mid+1, node.right)
            build(node.left_child, nums)
            build(node.right_child, nums)
            node.include = node.left_child.exclude + node.right_child.include
            node.exclude = max(node.left_child.include, node.left_child.exclude) + max(node.right_child.include, node.right_child.exclude)
        
        def update(node, index, value):
            if node.left == node.right:
                node.include = max(value, 0)
                node.exclude = 0
                return
            if index <= node.left_child.right:
                update(node.left_child, index, value)
            else:
                update(node.right_child, index, value)
            node.include = node.left_child.exclude + node.right_child.include
            node.exclude = max(node.left_child.include, node.left_child.exclude) + max(node.right_child.include, node.right_child.exclude)
        
        root = SegmentTreeNode(0, n-1)
        build(root, nums)
        
        total = 0
        for pos, x in queries:
            update(root, pos, x)
            total = (total + max(root.include, root.exclude)) % MOD
        return total