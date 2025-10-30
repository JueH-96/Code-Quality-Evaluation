import bisect

class Solution:
  def findMaximumLength(self, nums: list[int]) -> int:
    N = len(nums)
    
    # Prefix sums P: P[k] = sum(nums[0...k-1]), P[0] = 0
    P = [0] * (N + 1)
    for i in range(N):
      P[i+1] = P[i] + nums[i]

    # Segment tree setup
    # Operates on indices 0 to N (N+1 elements/leaves)
    # Stores pairs (f_val, P_val)
    # Comparison: maximize f_val, then maximize P_val
    num_leaves = N + 1 
    capacity = 1
    while capacity < num_leaves:
      capacity <<= 1
    
    # Identity for query: An (f, P) pair smaller than any valid pair.
    # f_val >= 0 (length), P_val >= 0 (sum).
    # (-1, -float('inf')) ensures any valid (f_val, P_val) pair is "greater".
    identity = (-1, -float('inf')) 
    seg_tree_data = [identity] * (2 * capacity)

    def merge_nodes(pair1, pair2):
      # Prioritize larger f_val
      if pair1[0] > pair2[0]:
        return pair1
      if pair2[0] > pair1[0]:
        return pair2
      # If f_vals are equal, prioritize larger P_val
      if pair1[1] > pair2[1]:
        return pair1
      else:
        return pair2

    def seg_tree_update(idx, val_pair):
      # idx is the original index (0 to N)
      pos = capacity + idx
      seg_tree_data[pos] = val_pair
      pos //= 2 # Move to parent
      while pos >= 1: # Root is at index 1
        seg_tree_data[pos] = merge_nodes(seg_tree_data[pos*2], seg_tree_data[pos*2+1])
        pos //= 2

    def seg_tree_query(L, R): # Query on original indices [L, R] inclusive
      res = identity
      L += capacity
      R += capacity
      while L <= R:
        if L % 2 == 1: # If L is a right child, use its value and advance L
          res = merge_nodes(res, seg_tree_data[L])
          L += 1
        if R % 2 == 0: # If R is a left child, use its value and advance R
          res = merge_nodes(res, seg_tree_data[R])
          R -= 1
        # Move to parent level
        L //= 2
        R //= 2
      return res

    # X_coords stores g[j]+P[j] values. This list is proven to be non-decreasing.
    # X_coords[j_original_idx] = g[j_original_idx] + P[j_original_idx]
    X_coords = [] 

    # Base case for k=0 (representing an empty prefix before nums[0])
    # f[0]: max length is 0.
    # P[0]: sum of empty prefix is 0.
    # g[0]: sum of last segment (non-existent) is 0. This allows the first segment to be anything.
    f0 = 0
    P0 = P[0] # This is 0
    g0 = 0 
    
    seg_tree_update(0, (f0, P0)) # Store (f[0], P[0]) at index 0 in segment tree
    X_coords.append(g0 + P0)     # X_coords[0] = g[0]+P[0]
    
    # Current answer, will be updated iteratively.
    # Constraints: 1 <= nums.length, so N >= 1.
    # The loop runs from k=1 to N. ans_f will be fk for k=N.
    current_f_val = 0 

    for k in range(1, N + 1):
      pk = P[k] # Prefix sum up to nums[k-1]
      
      # Find j_max: largest original index j such that g[j]+P[j] <= pk.
      # This means X_coords[j] <= pk.
      # bisect_right returns an insertion point. `idx_insert_point - 1` gives j_max.
      idx_insert_point = bisect.bisect_right(X_coords, pk)
      j_max = idx_insert_point - 1
      
      # Query segment tree for max (f_val, P_val) in original index range [0, j_max].
      # This finds the optimal (f[j_optimal], P[j_optimal]).
      opt_f, opt_P = seg_tree_query(0, j_max)
      
      # Calculate f[k] and g[k]
      fk = opt_f + 1
      gk = pk - opt_P # Sum of the current segment nums[j_optimal ... k-1]
      
      current_f_val = fk # Update current answer
      
      # Store dp[k] information for future steps
      seg_tree_update(k, (fk, P[k]))  # Store (f[k], P[k]) for original index k
      X_coords.append(gk + pk)       # X_coords[k] = g[k]+P[k]
      
    return current_f_val