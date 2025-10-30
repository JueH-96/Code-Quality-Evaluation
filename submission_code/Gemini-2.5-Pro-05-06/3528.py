from typing import List

# Using a very small integer for negative infinity to keep all calculations with integers.
_NEGINF_INT = -(10**18 + 7) 

class _Line:
    def __init__(self, m: int, c: int):
        self.m = m
        self.c = c

    def eval(self, x: int) -> int: 
        # Check for the specific default line representation
        if self.m == 0 and self.c == _NEGINF_INT: 
            return _NEGINF_INT
        return self.m * x + self.c 

_DEFAULT_LINE = _Line(0, _NEGINF_INT)

class _LiChaoTree:
    def __init__(self, num_coords: int):
        # num_coords is n, the number of indices. Tree operates on x-coordinates [0, n-1].
        self.num_coords = num_coords
        
        # Smallest power of 2 greater than or equal to num_coords
        self.k = 1
        while self.k < self.num_coords:
            self.k *= 2
        
        # 1-indexed segment tree array stores lines.
        # Size 2*k is sufficient for a segment tree that covers [0, k-1].
        self.tree_lines = [_DEFAULT_LINE] * (2 * self.k)

    def add_line(self, new_line: _Line):
        # Add line to the tree. Root is node_idx=1, covers range [0, k-1].
        self._add_line_recursive(1, new_line, 0, self.k - 1)

    def _add_line_recursive(self, node_idx: int, new_line: _Line, r_low: int, r_high: int):
        # r_low, r_high: coordinate range covered by this node.
        current_line_in_node = self.tree_lines[node_idx]
        r_mid = (r_low + r_high) // 2
        
        # Determine which line is better at the midpoint of the current range.
        new_is_better_at_mid = new_line.eval(r_mid) > current_line_in_node.eval(r_mid)
        
        if new_is_better_at_mid:
            # new_line is better at r_mid. It becomes this node's primary line.
            # The old line (current_line_in_node) is passed down.
            self.tree_lines[node_idx], new_line = new_line, current_line_in_node
        
        # Now, self.tree_lines[node_idx] is the line better at r_mid.
        # new_line is worse (or equal) at r_mid. It might still be optimal in a sub-range.
        
        # If new_line is now the _DEFAULT_LINE, it cannot improve any score, so stop.
        if new_line.m == _DEFAULT_LINE.m and new_line.c == _DEFAULT_LINE.c:
            return
            
        if r_low == r_high: # Leaf node, no further recursion.
            return

        # If new_line (worse at r_mid) is better at r_low than the current node's line:
        # This means their intersection is in the left half [r_low, r_mid).
        # So, new_line might be optimal in parts of the left child's range.
        if new_line.eval(r_low) > self.tree_lines[node_idx].eval(r_low):
            self._add_line_recursive(2 * node_idx, new_line, r_low, r_mid)
        # Else, if new_line is better at r_high:
        # Intersection is in the right half (r_mid, r_high].
        # new_line might be optimal in parts of the right child's range.
        # Use elif: a line intersects another at most once. If it dominates in one part
        # (e.g., new_line better at r_low), it won't "cross back" to dominate in the other part
        # if it's already worse at r_mid.
        elif new_line.eval(r_high) > self.tree_lines[node_idx].eval(r_high):
            self._add_line_recursive(2 * node_idx + 1, new_line, r_mid + 1, r_high)

    def query(self, x: int) -> int:
        # Query for the max value at coordinate x. Root is node_idx=1, covers [0, k-1].
        return self._query_recursive(1, x, 0, self.k - 1)

    def _query_recursive(self, node_idx: int, x: int, r_low: int, r_high: int) -> int:
        # Max value from the line stored at this node.
        max_val = self.tree_lines[node_idx].eval(x)
        
        r_mid = (r_low + r_high) // 2
        
        if r_low == r_high: # Leaf node.
            return max_val
            
        # Recursively query the appropriate child.
        if x <= r_mid:
            # Check bounds: 2*node_idx must be < len(self.tree_lines). Safe with 1-based indexing and 2*k size.
            max_val = max(max_val, self._query_recursive(2 * node_idx, x, r_low, r_mid))
        else: # x > r_mid
            max_val = max(max_val, self._query_recursive(2 * node_idx + 1, x, r_mid + 1, r_high))
        
        return max_val

class Solution:
  def findMaximumScore(self, nums: List[int]) -> int:
    n = len(nums)
    if n == 1: # Start at index 0, target is index 0. No jumps needed. Score = 0.
        return 0

    # dp_val_at_j will store dp[j] as we iterate.
    # Initialize with dp[0] = 0. This variable will be updated in the loop.
    dp_val_at_j = 0 

    # Li Chao Tree for coordinates 0 to n-1.
    lct = _LiChaoTree(n) 

    # Base case: dp[0] = 0.
    # The line corresponding to index 0 is: m_0 = nums[0], c_0 = dp[0] - 0*nums[0] = 0.
    # So, y = nums[0]*x.
    line0 = _Line(nums[0], 0) 
    lct.add_line(line0)

    # Iterate j from 1 to n-1 to calculate dp[j].
    for j in range(1, n):
        # Query the Li Chao Tree at x = j to find max_{i<j} (dp[i] + (j-i)*nums[i]).
        # This value is dp[j].
        max_score_to_reach_j = lct.query(j)
        dp_val_at_j = max_score_to_reach_j
                                                    
        # If j is not the last index, add the line for index j to the tree.
        # This line L_j (m_j=nums[j], c_j=dp[j]-j*nums[j]) will be used for future queries k > j.
        # Line L_{n-1} is not needed as dp[n-1] is the final answer.
        if j < n - 1: 
            line_j = _Line(nums[j], dp_val_at_j - j * nums[j])
            lct.add_line(line_j)
            
    # After the loop, dp_val_at_j holds dp[n-1].
    return dp_val_at_j