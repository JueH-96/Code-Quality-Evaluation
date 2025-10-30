# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        total = [0]
        curr = 0
        
        def dfs(root, curr):
            if not root: return
            curr = curr ^ (1 << root.val)
            dfs(root.left, curr)
            dfs(root.right, curr)
            if not root.left and not root.right:
                total[0] += (curr & (curr - 1) == 0 or curr & (curr - 1) & (curr - 2) == 0)
        dfs(root, curr)
        return total[0]