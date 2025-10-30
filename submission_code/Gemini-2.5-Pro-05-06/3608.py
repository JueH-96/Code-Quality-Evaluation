import math

class Solution:
  def subsequencePairCount(self, nums: list[int]) -> int:
    MOD = 10**9 + 7
    
    if not nums:
      return 0
      
    max_val_in_nums = 0
    # Find maximum value in nums to set bounds for GCD table and G iteration
    for x in nums:
      if x > max_val_in_nums:
        max_val_in_nums = x

    # Precompute GCD values
    # gcd_table[a][b] stores gcd(a, b)
    # math.gcd(0, k) = k, math.gcd(0, 0) = 0.
    gcd_table = [[0] * (max_val_in_nums + 1) for _ in range(max_val_in_nums + 1)]
    for i in range(max_val_in_nums + 1):
      for j in range(i, max_val_in_nums + 1): 
        val = math.gcd(i, j)
        gcd_table[i][j] = val
        gcd_table[j][i] = val

    total_ans = 0
    
    # Iterate over all possible common GCD values G_val
    for G_val in range(1, max_val_in_nums + 1):
      # S_G: list of (num / G_val) for nums that are multiples of G_val
      S_G = []
      current_max_val_in_S_G = 0 # Max value in the current S_G list
      for x_num in nums:
        if x_num % G_val == 0:
          val_in_S_G = x_num // G_val
          S_G.append(val_in_S_G)
          if val_in_S_G > current_max_val_in_S_G:
            current_max_val_in_S_G = val_in_S_G
      
      if not S_G: # If no elements are multiples of G_val, this G_val is not possible
        continue

      # Initialize DP table for this G_val
      # dp_current_iter[g1][g2] = count of ways
      dp_current_iter = [[0] * (current_max_val_in_S_G + 1) for _ in range(current_max_val_in_S_G + 1)]
      dp_current_iter[0][0] = 1 # Base case: two empty subsequences, 1 way

      # Process each element from S_G
      for x_sg_elem in S_G:
        # dp_next_iter stores states after considering x_sg_elem
        # Initialize by copying dp_current_iter: accounts for x_sg_elem not being used
        dp_next_iter = [row[:] for row in dp_current_iter]
        
        for g1 in range(current_max_val_in_S_G + 1): # Previous GCD for seq1'
          for g2 in range(current_max_val_in_S_G + 1): # Previous GCD for seq2'
            val_count = dp_current_iter[g1][g2] # Ways to reach state (g1,g2) before x_sg_elem
            if val_count == 0:
              continue
            
            # Option 2: x_sg_elem is added to seq1'
            # New GCD for seq1' will be gcd(g1, x_sg_elem).
            # (If g1 was 0 (seq1' empty), new GCD is x_sg_elem, as gcd_table[0][k]=k)
            next_g1 = gcd_table[g1][x_sg_elem]
            dp_next_iter[next_g1][g2] = (dp_next_iter[next_g1][g2] + val_count) % MOD
            
            # Option 3: x_sg_elem is added to seq2'
            next_g2 = gcd_table[g2][x_sg_elem]
            dp_next_iter[g1][next_g2] = (dp_next_iter[g1][next_g2] + val_count) % MOD
            
        dp_current_iter = dp_next_iter # Move to next state table
      
      # After processing all elements in S_G for current G_val:
      # dp_current_iter[1][1] is the count of pairs (seq1', seq2') from S_G
      # such that GCD(seq1') = 1 and GCD(seq2') = 1.
      # This means original subsequences seq1, seq2 from nums have
      # GCD(seq1) = G_val and GCD(seq2) = G_val.
      # A GCD of 1 implies the subsequence is non-empty.
      ans_for_G = dp_current_iter[1][1]
      total_ans = (total_ans + ans_for_G) % MOD
      
    return total_ans