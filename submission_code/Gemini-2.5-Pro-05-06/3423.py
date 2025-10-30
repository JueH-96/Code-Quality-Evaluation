import math
from typing import List

class Solution:
  def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
    MOD = 10**9 + 7
    N = len(nums)
    
    # Constraints: 1 <= nums.length. So N >= 1.
    # No need to handle N=0 explicitly based on constraints.

    neg_inf = -float('inf')

    # Segment tree array. Size 4*N for 1-indexed recursive implementations.
    # tree[node_idx] stores a 2x2 matrix as a flat list [T00, T01, T10, T11]
    tree = [[neg_inf] * 4 for _ in range(4 * N)]

    # Matrix multiplication: C_matrix = A_matrix @ B_matrix
    # A_matrix is for the right segment, B_matrix is for the left segment.
    # (Transformation S_final = A_matrix @ (B_matrix @ S_initial) = (A_matrix @ B_matrix) @ S_initial)
    def multiply_matrices(A_matrix, B_matrix):
      a00, a01, a10, a11 = A_matrix[0], A_matrix[1], A_matrix[2], A_matrix[3]
      b00, b01, b10, b11 = B_matrix[0], B_matrix[1], B_matrix[2], B_matrix[3]
      
      # C_ij = max_k (A_ik + B_kj)
      C00 = max(a00 + b00, a01 + b10) 
      C01 = max(a00 + b01, a01 + b11)
      C10 = max(a10 + b00, a11 + b10)
      C11 = max(a10 + b01, a11 + b11)
      return [C00, C01, C10, C11]

    # Build segment tree from initial nums array
    def build(node_idx, L, R):
      if L == R:
        # Leaf node matrix for nums[L]: [[0, 0], [nums[L], -inf]]
        tree[node_idx] = [0, 0, nums[L], neg_inf]
        return
      
      M = (L + R) // 2
      build(2 * node_idx, L, M) # Left child
      build(2 * node_idx + 1, M + 1, R) # Right child
      
      # Parent matrix = Right_child_matrix @ Left_child_matrix
      tree[node_idx] = multiply_matrices(tree[2 * node_idx + 1], tree[2 * node_idx])

    # Update value at target_idx to val
    def update(target_idx, val, node_idx, L, R):
      if L == R: # Leaf node for target_idx
        tree[node_idx] = [0, 0, val, neg_inf]
        return

      M = (L + R) // 2
      if target_idx <= M:
        update(target_idx, val, 2 * node_idx, L, M)
      else:
        update(target_idx, val, 2 * node_idx + 1, M + 1, R)
      
      tree[node_idx] = multiply_matrices(tree[2 * node_idx + 1], tree[2 * node_idx])
        
    build(1, 0, N - 1)
    
    total_max_sum = 0
    
    for pos, x_val in queries:
      update(pos, x_val, 1, 0, N - 1)
      
      # Root matrix T_root for range [0, N-1] is tree[1]
      T_root = tree[1]
      
      # Apply T_root to initial state S_{-1} = [0, -inf]^T
      # Resulting state S_{N-1} = [dp[N-1][0], dp[N-1][1]]
      # dp[N-1][0] = max(T_root_00 + S_{-1,0}, T_root_01 + S_{-1,1})
      #            = max(T_root[0] + 0,       T_root[1] + neg_inf)
      # dp[N-1][1] = max(T_root_10 + S_{-1,0}, T_root_11 + S_{-1,1})
      #            = max(T_root[2] + 0,       T_root[3] + neg_inf)
      
      dp_N_exclude = max(T_root[0], T_root[1] + neg_inf) # T_root[0] is effectively T_root[0]+0
      dp_N_include = max(T_root[2], T_root[3] + neg_inf) # T_root[2] is effectively T_root[2]+0

      # Max sum is max(0 for empty subseq, dp_N_exclude, dp_N_include)
      # Python's max(0, -float('inf')) correctly returns 0.
      current_ans = max(0, dp_N_exclude, dp_N_include)
      
      total_max_sum = (total_max_sum + current_ans) % MOD
      
    return int(total_max_sum) # Cast to int just in case float arithmetic made total_max_sum float