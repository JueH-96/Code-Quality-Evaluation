import math
from typing import List
# import sys
# Set higher recursion depth if needed for very large N, though N=10^5 should be okay.
# sys.setrecursionlimit(200000) 

# Define a sufficiently small integer value to represent negative infinity.
# This avoids potential floating point precision issues with math.inf.
# Python integers handle arbitrary size, so 10**18 is well within limits.
NEG_INF = -(10**18) 

# Helper function to evaluate a line y = m*x + c at a given point x.
# Uses integer arithmetic.
def eval_line(line, x):
    """Calculates the value of the line (m, c) at point x."""
    m, c = line
    # If the intercept is the sentinel NEG_INF, the value is NEG_INF.
    if c == NEG_INF: 
        return NEG_INF
    # Standard line evaluation.
    return m * x + c

# Node class for the Li Chao Tree.
class Node:
    """Represents a node in the Li Chao Tree."""
    def __init__(self):
        # Each node stores the line (slope m, intercept c) that is maximal
        # over some part of the node's interval.
        # Initialize with a line representing NEG_INF score.
        self.line = (0, NEG_INF) 
        self.left = None  # Left child node
        self.right = None # Right child node

# Li Chao Tree implementation for finding the maximum value of a set of linear functions.
class LiChaoTree:
    """Implements a Li Chao Tree for maximizing linear functions."""
    def __init__(self, N):
        """Initializes the tree for the coordinate range [0, N-1]."""
        self.root = Node()
        # N is the size of the input array `nums`. Indices range from 0 to N-1.
        # The tree covers the coordinate range [0, N-1].
        self.N = N 

    def insert(self, line):
        """Inserts a new line (m, c) into the tree."""
        # Calls the recursive helper function starting from the root.
        self._insert(self.root, 0, self.N - 1, line)

    def _insert(self, node, L, R, new_line):
        """Recursively inserts 'new_line' into the subtree rooted at 'node'."""
        
        if node is None: # Safety check, should not be needed if nodes created properly
             return 

        # Midpoint of the current interval [L, R]
        mid = (L + R) // 2
        
        # Evaluate both the new line and the node's current line at the midpoint.
        new_line_val_mid = eval_line(new_line, mid)
        node_line_val_mid = eval_line(node.line, mid)

        # Determine if the new line gives a higher value at 'mid'.
        is_new_better_at_mid = new_line_val_mid > node_line_val_mid
        
        if is_new_better_at_mid:
            # If the new line is better at mid, it becomes this node's line.
            # The old line stored at the node might still be maximal in a sub-interval,
            # so it is potentially pushed down ('new_line' now refers to the old line).
            node.line, new_line = new_line, node.line

        # If the interval [L, R] represents more than one point (L < R),
        # the line that was worse at 'mid' ('new_line' after potential swap)
        # might still be the maximal line in either the left or right sub-interval.
        if L < R:
            # Evaluate the line 'new_line' (the one worse at mid) at the interval boundaries.
            new_line_val_L = eval_line(new_line, L)
            node_line_val_L = eval_line(node.line, L)
            new_line_val_R = eval_line(new_line, R)
            node_line_val_R = eval_line(node.line, R)
            
            # Only proceed if 'new_line' is not the initial sentinel line (intercept NEG_INF).
            # This prevents propagating the sentinel line unnecessarily.
            if new_line[1] != NEG_INF:
                 # Check if 'new_line' is better at the left boundary L.
                 if new_line_val_L > node_line_val_L:
                     # If yes, it might be maximal in the left sub-interval [L, mid].
                     # Recurse into the left child. Create node if it doesn't exist.
                     if node.left is None: node.left = Node()
                     self._insert(node.left, L, mid, new_line)
                 # Else, check if 'new_line' is better at the right boundary R.
                 # The 'elif' is key: a line worse at 'mid' can cross the dominant line
                 # at most once within [L, R], so it can be better in at most one half.
                 elif new_line_val_R > node_line_val_R:
                     # If yes, it might be maximal in the right sub-interval [mid+1, R].
                     # Recurse into the right child. Create node if it doesn't exist.
                     if node.right is None: node.right = Node()
                     self._insert(node.right, mid + 1, R, new_line)

    def query(self, x):
        """Queries the maximum value among all inserted lines at coordinate x."""
        # Calls the recursive helper function starting from the root.
        return self._query(self.root, 0, self.N - 1, x)

    def _query(self, node, L, R, x):
        """Recursively queries the maximum line value at point x in the subtree."""
        
        # If we reach a non-existent node (path doesn't exist), return NEG_INF.
        if node is None:
            return NEG_INF 

        # Midpoint of the current interval [L, R]
        mid = (L + R) // 2
        # The maximum value is at least the value from the line stored at the current node.
        max_val = eval_line(node.line, x)

        # If the current interval is not a single point, recursively query the child
        # corresponding to the interval containing x.
        if L < R:
            if x <= mid:
                # If x is in the left half [L, mid], query the left child.
                max_val = max(max_val, self._query(node.left, L, mid, x))
            else:
                # If x is in the right half [mid+1, R], query the right child.
                max_val = max(max_val, self._query(node.right, mid + 1, R, x))
        
        # Return the maximum value found along the path from the root to the leaf for x.
        return max_val

# Main Solution Class using Dynamic Programming with Li Chao Tree optimization.
class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        """
        Calculates the maximum score to reach the end of the array `nums`.
        Uses Dynamic Programming optimized with a Li Chao Tree.
        """
        n = len(nums)
        
        # Base case: If there's only one element, we start at index 0 and are already
        # at the destination (index n-1 = 0). The score is 0.
        if n == 1:
            return 0
        
        # dp[j] stores the maximum score to reach index j.
        # Initialized with 0s. dp[0] = 0 is the base case score for starting point.
        dp = [0] * n 
        
        # Initialize the Li Chao Tree. It will manage lines representing potential scores.
        # The coordinate range corresponds to the array indices [0, n-1].
        tree = LiChaoTree(n)

        # Process the starting index 0.
        # The score to reach index 0 is dp[0] = 0.
        # The corresponding line L_0 for potential jumps from index 0 is:
        # L_0(x) = nums[0]*x + (dp[0] - 0*nums[0]) 
        #        = nums[0]*x + 0
        m0 = nums[0]
        c0 = dp[0] - 0 * nums[0] # This simplifies to 0
        tree.insert((m0, c0))
        
        # Iterate through the array indices j from 1 up to n-1.
        for j in range(1, n):
            # Query the Li Chao Tree at coordinate x=j. This finds the maximum value of
            # L_i(j) = nums[i]*j + (dp[i] - i*nums[i]) over all previously processed indices i < j.
            # This corresponds to max_{0 <= i < j} (dp[i] + (j-i)*nums[i]).
            current_max_score = tree.query(j)
            
            # dp[j] is the maximum score found to reach index j.
            # Since all jump scores (j-i)*nums[i] are positive for j>i and nums[i]>=1,
            # and dp[0]=0, all subsequent dp[j] values must be non-negative.
            # The query should not return NEG_INF unless n=1 (handled) or an error occurs.
            dp[j] = current_max_score
            
            # If index j is not the last index (n-1), we need to potentially jump from it later.
            # Calculate the line L_j corresponding to reaching index j with score dp[j]
            # and insert it into the tree for future queries (for k > j).
            # L_j(x) = nums[j]*x + (dp[j] - j*nums[j])
            if j < n - 1:
                mj = nums[j]
                # Calculate the intercept cj for line L_j.
                cj = dp[j] - j * nums[j]
                tree.insert((mj, cj))
                
        # The final answer is the maximum score to reach the last index, n-1.
        return dp[n-1]