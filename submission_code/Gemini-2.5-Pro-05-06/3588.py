class Solution:
  def countWinningSequences(self, s: str) -> int:
    n = len(s)
    MOD = 10**9 + 7

    # Mapping characters to codes: 'F':0, 'W':1, 'E':2
    # Sentinel for previous move when i=0 (no previous move): 3
    alice_moves_codes = []
    for char_a in s:
      if char_a == 'F':
        alice_moves_codes.append(0)
      elif char_a == 'W':
        alice_moves_codes.append(1)
      else: # char_a == 'E'
        alice_moves_codes.append(2)

    # Precompute powers of 2 modulo MOD
    powers_of_2 = [1] * (n + 1)
    for k in range(1, n + 1):
      powers_of_2[k] = (powers_of_2[k-1] * 2) % MOD
    
    # dp[i][prev_B_idx][score_sum_offset] stores the number of ways.
    # i: current round index, from 0 to n.
    # prev_B_idx: Bob's move in round i-1 (0,1,2 for F,W,E; 3 for initial dummy state at i=0).
    # score_sum_offset: (Alice's score - Bob's score) + n. Ranges from 0 to 2*n.
    
    dp = [[[0] * (2 * n + 1) for _ in range(4)] for _ in range(n + 1)]

    # Base case: i = n (all rounds played)
    # Bob wins if actual_score_diff < 0, which means score_sum_offset < n.
    for prev_B_idx_val in range(4): # 0,1,2 for F,W,E; 3 for dummy state
      for score_offset_val in range(n): # score_offset_val from 0 to n-1
        dp[n][prev_B_idx_val][score_offset_val] = 1
    # For score_offset_val >= n, it's 0, which is the default initialization.

    # Iterate i from n-1 down to 0
    for i in range(n - 1, -1, -1):
      A_char_code = alice_moves_codes[i] # Alice's move in current round i
      num_rounds_left = n - i # Number of rounds from i to n-1

      # Values for Pruning 2 (Bob guaranteed to win)
      # Case 1: Current round i is the first round of the game (i=0).
      # Bob's prev_B_idx is 3 (sentinel). num_rounds_left is n.
      ways_guaranteed_win_if_i_is_0 = 0
      if num_rounds_left == 1: # n=1, Bob has 3 choices for this single round
        ways_guaranteed_win_if_i_is_0 = 3
      elif num_rounds_left > 1: # n > 1
        ways_guaranteed_win_if_i_is_0 = (3 * powers_of_2[num_rounds_left - 1]) % MOD
      
      # Case 2: Current round i is not the first (i > 0).
      # Bob's prev_B_idx is 0, 1, or 2.
      ways_guaranteed_win_if_i_is_not_0 = 0
      if num_rounds_left >= 1: # Bob has 2 choices for round i, then 2 for each of remaining rounds.
        ways_guaranteed_win_if_i_is_not_0 = powers_of_2[num_rounds_left]
      # if num_rounds_left == 0, this means i=n, handled by base case.
      
      # Determine which prev_B_idx values and score_sum_offsets to iterate for current i.
      # Loop for prev_B_idx:
      # If i == 0, prev_B_idx is 3 (sentinel). Score_sum_offset must be n (initial score diff 0).
      # If i > 0, prev_B_idx is 0,1,2. Score_sum_offset is in [n-i, n+i].
      
      prev_B_indices_to_process = []
      if i == 0:
        prev_B_indices_to_process.append(3)
      else:
        prev_B_indices_to_process.extend([0, 1, 2])

      for prev_B_idx in prev_B_indices_to_process:
        score_offsets_to_process = []
        if i == 0: # And prev_B_idx must be 3
            score_offsets_to_process.append(n) # Initial score offset
        else: # i > 0, prev_B_idx in {0,1,2}
            # Valid actual score diff for rounds 0..i-1 is [-i, i]. Offsetted: [n-i, n+i].
            # Iterate only through reachable score offsets.
            min_s_offset = n - i
            max_s_offset = n + i
            for s_offset in range(min_s_offset, max_s_offset + 1):
                 score_offsets_to_process.append(s_offset)
        
        for score_sum_offset in score_offsets_to_process:
          # Pruning 1: Bob cannot win.
          # current actual score_diff = score_sum_offset - n
          # Bob wins all 'num_rounds_left', gains 'num_rounds_left' points (score_diff decreases by num_rounds_left).
          # Smallest final score_diff = (score_sum_offset - n) - num_rounds_left.
          # If this is >= 0, Bob cannot win.
          if (score_sum_offset - n) - num_rounds_left >= 0:
            dp[i][prev_B_idx][score_sum_offset] = 0
            continue

          # Pruning 2: Bob guaranteed to win.
          # Alice wins all 'num_rounds_left', gains 'num_rounds_left' points (score_diff increases by num_rounds_left).
          # Largest final score_diff = (score_sum_offset - n) + num_rounds_left.
          # If this is < 0, Bob is guaranteed to win.
          if (score_sum_offset - n) + num_rounds_left < 0:
            if prev_B_idx == 3: # This implies i == 0
              dp[i][prev_B_idx][score_sum_offset] = ways_guaranteed_win_if_i_is_0
            else: # i > 0
              dp[i][prev_B_idx][score_sum_offset] = ways_guaranteed_win_if_i_is_not_0
            continue
          
          # Standard DP calculation if not pruned
          current_total_ways = 0
          
          # Determine Bob's options for current move B_char_code for round i
          possible_B_moves_for_round_i = []
          if prev_B_idx == 3: # Bob's first move (round 0 has no preceding Bob move)
            possible_B_moves_for_round_i = [0, 1, 2]
          else: # Bob's subsequent moves (round i > 0)
            for move_candidate in [0, 1, 2]:
              if move_candidate != prev_B_idx: # Cannot repeat previous move
                possible_B_moves_for_round_i.append(move_candidate)
          
          for B_char_code in possible_B_moves_for_round_i: # Bob's move for round i
            score_delta = 0 # Change in (Alice's score - Bob's score)
            if A_char_code == B_char_code: # Same creature
              score_delta = 0
            # Alice wins if (A_char_code - B_char_code + 3) % 3 == 1
            elif (A_char_code - B_char_code + 3) % 3 == 1: 
              score_delta = 1
            else: # Bob wins, (B_char_code - A_char_code + 3) % 3 == 1
              score_delta = -1
            
            new_score_sum_offset = score_sum_offset + score_delta
            
            # Add ways from next state. new_score_sum_offset will be within [0, 2*n].
            # dp[i+1] states consider prev_B_idx = B_char_code (Bob's move in round i)
            # and score_sum_offset = new_score_sum_offset (accumulated till end of round i).
            current_total_ways = (current_total_ways + dp[i+1][B_char_code][new_score_sum_offset]) % MOD
          
          dp[i][prev_B_idx][score_sum_offset] = current_total_ways
          
    # Final answer is for i=0, prev_B_idx=3 (no prior move), score_sum_offset=n (initial score 0)
    return dp[0][3][n]