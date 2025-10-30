import collections

class Solution:
  def maxValue(self, nums: list[int], k: int) -> int:
    n = len(nums)
    
    # L[i][j] stores the set of OR sums of subsequences of length j
    # where nums[i] is the j-th (last) element.
    L = [[set() for _ in range(k + 1)] for _ in range(n)]
    
    # L_prefix_union[i][j] stores the union of L[x][j] for x from 0 to i.
    # This means it's the set of OR sums of subsequences of length j
    # that end at any index p, where p <= i.
    L_prefix_union = [[set() for _ in range(k + 1)] for _ in range(n)]

    for i in range(n):
      # Base case: subsequence of length 1 ending at nums[i].
      # The OR sum is just nums[i] itself.
      L[i][1] = {nums[i]}
      
      # For subsequences of length j (from 2 to k) ending at nums[i]:
      # The (j-1)-th element must be some nums[x] where x < i.
      # The OR sum of the first j-1 elements can be any value in L_prefix_union[i-1][j-1].
      if i > 0: # Must have at least one element before nums[i] to form longer subsequences.
        for j in range(2, k + 1):
          # If L_prefix_union[i-1][j-1] is empty, it means no subsequence of 
          # length j-1 can be formed ending before nums[i].
          if not L_prefix_union[i-1][j-1]:
              continue
          for prev_or_sum in L_prefix_union[i-1][j-1]:
            L[i][j].add(prev_or_sum | nums[i])
            
      # Update L_prefix_union table for current index i.
      # L_prefix_union[i][j] includes all OR sums from L[p][j] for p <= i.
      for j in range(1, k + 1):
        if i == 0:
          L_prefix_union[i][j] = L[i][j].copy() 
        else:
          L_prefix_union[i][j] = L_prefix_union[i-1][j].copy()
          L_prefix_union[i][j].update(L[i][j])

    # R[i][j] stores the set of OR sums of subsequences of length j
    # where nums[i] is the 1st element.
    R = [[set() for _ in range(k + 1)] for _ in range(n)]
    
    # R_suffix_union[i][j] stores the union of R[x][j] for x from i to n-1.
    # This means it's the set of OR sums of subsequences of length j
    # that start at any index p, where p >= i.
    R_suffix_union = [[set() for _ in range(k + 1)] for _ in range(n)]

    for i in range(n - 1, -1, -1):
      # Base case: subsequence of length 1 starting at nums[i].
      R[i][1] = {nums[i]}
      
      # For subsequences of length j (from 2 to k) starting at nums[i]:
      # nums[i] is the first element. The other j-1 elements form a subsequence
      # that must start at some nums[x] where x > i.
      # The OR sum of these j-1 elements can be any value in R_suffix_union[i+1][j-1].
      if i < n - 1: # Must have at least one element after nums[i].
        for j in range(2, k + 1):
          if not R_suffix_union[i+1][j-1]:
              continue
          for suffix_or_sum in R_suffix_union[i+1][j-1]:
            R[i][j].add(suffix_or_sum | nums[i])
            
      # Update R_suffix_union table for current index i.
      for j in range(1, k + 1):
        if i == n - 1:
          R_suffix_union[i][j] = R[i][j].copy()
        else:
          R_suffix_union[i][j] = R_suffix_union[i+1][j].copy()
          R_suffix_union[i][j].update(R[i][j])

    max_xor_val = 0
    
    # Iterate over all possible split points.
    # nums[i] is the last element of the first group (length k).
    # The first group's OR sum is in L[i][k].
    # The second group (length k) must start at an index p > i.
    # The set of OR sums for such second groups is R_suffix_union[i+1][k].
    
    min_idx_first_group_ends = k - 1 
    max_idx_first_group_ends = n - 1 - k 
    # Loop range for i ensures there are enough elements for both groups.
    # Smallest i means first group is nums[0...k-1].
    # Largest i means first group ends at nums[n-1-k], so second group is nums[n-k...n-1].
    # If min_idx_first_group_ends > max_idx_first_group_ends, n < 2k,
    # which is ruled out by problem constraint k <= nums.length / 2.

    for i in range(min_idx_first_group_ends, max_idx_first_group_ends + 1):
      set_A = L[i][k] 
      if not set_A: # No way to form the first group ending at nums[i]
        continue
      
      # Second group must start at index i+1 or later.
      # R_suffix_union[i+1][k] contains OR sums of all k-length subsequences
      # starting at an index p >= i+1.
      set_B = R_suffix_union[i+1][k]
      if not set_B: # No way to form the second group starting after nums[i]
        continue
        
      # Combine OR sums from both groups
      for orA in set_A:
        for orB in set_B:
          max_xor_val = max(max_xor_val, orA ^ orB)
          
    return max_xor_val