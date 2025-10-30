import math

class Solution:
  def minimumChanges(self, s: str, k: int) -> int:
    n = len(s)

    # Precompute divisors for all possible lengths
    # divs[length] will store list of divisors d such that 1 <= d < length
    divs = [[] for _ in range(n + 1)]
    # O(N log N) way to compute divisors:
    # For each d_val, it's a divisor of multiples d_val*m.
    # We need d_val < length (i.e. d_val < d_val*m), so m > 1.
    # Thus, length_val starts from 2*d_val.
    for d_val in range(1, n + 1): 
        for length_val in range(2 * d_val, n + 1, d_val):
            divs[length_val].append(d_val)
    
    # min_substring_cost[i][j] = min changes to make s[i..j] a semi-palindrome
    # Initialize with a large value (n + 1 is fine as max changes for a substring is n/2)
    min_substring_cost = [[n + 1] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n): # Iterate over all substrings s[i..j]
            sub_len = j - i + 1
            if sub_len < 2: # Substrings must have length at least 2 to be semi-palindrome
                # min_substring_cost[i][j] remains n+1 (infinity)
                continue

            current_min_for_this_substring = n + 1 # Cost for s[i..j]
            
            # Iterate over all valid d for s[i..j]
            # divs[sub_len] contains d such that d | sub_len and d < sub_len.
            # If sub_len is prime and > 1, divs[sub_len] will contain [1].
            # If sub_len = 1, divs[sub_len] is empty (but we skip sub_len < 2).
            for d in divs[sub_len]:
                num_blocks = sub_len // d
                changes_for_this_d = 0
                
                # Compare characters based on block structure
                # B_block_idx vs B_{num_blocks-1-block_idx}
                # For char_offset_in_block k: s[i + block_idx*d + k] vs s[i + (num_blocks-1-block_idx)*d + k]
                for char_offset_in_block in range(d): 
                    for block_idx in range(num_blocks // 2): # Iterate up to the middle pair of blocks
                        idx1 = i + block_idx * d + char_offset_in_block
                        idx2 = i + (num_blocks - 1 - block_idx) * d + char_offset_in_block
                        if s[idx1] != s[idx2]:
                            changes_for_this_d += 1
                
                current_min_for_this_substring = min(current_min_for_this_substring, changes_for_this_d)
            
            min_substring_cost[i][j] = current_min_for_this_substring

    # Dynamic Programming
    # dp[len_pref][num_p] = min changes to partition s[0...len_pref-1] (length len_pref) 
    #                      into num_p substrings
    # Max changes for string s is n. So n+1 is a good infinity.
    dp = [[n + 1] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0 # 0 changes for empty string with 0 partitions

    for num_partitions in range(1, k + 1):
        for current_len_prefix in range(1, n + 1):
            # Min length for num_partitions substrings, each of length at least 2, is 2 * num_partitions
            if current_len_prefix < 2 * num_partitions:
                continue # Not possible to form num_partitions with this prefix length

            if num_partitions == 1:
                # The single substring is s[0...current_len_prefix-1]
                # Its length must be >= 2. This is ensured by current_len_prefix >= 2 * num_partitions.
                dp[current_len_prefix][1] = min_substring_cost[0][current_len_prefix-1]
            else:
                # The last substring is s[prev_prefix_len ... current_len_prefix-1]
                # prev_prefix_len is the length of the prefix covered by the first (num_partitions-1) substrings.
                
                # Min prev_prefix_len is 2 * (num_partitions - 1)
                min_prev_len = 2 * (num_partitions - 1)
                # Max prev_prefix_len is current_len_prefix - 2 (to leave length 2 for the last substring)
                max_prev_len = current_len_prefix - 2
                
                for prev_prefix_len in range(min_prev_len, max_prev_len + 1):
                    cost_of_last_substring = min_substring_cost[prev_prefix_len][current_len_prefix-1]
                    
                    if dp[prev_prefix_len][num_partitions-1] < n + 1: # Check previous state is reachable
                        # cost_of_last_substring is guaranteed to be < n+1 if its length >= 2,
                        # which is true because prev_prefix_len <= current_len_prefix - 2.
                        dp[current_len_prefix][num_partitions] = min(dp[current_len_prefix][num_partitions],
                                                                  dp[prev_prefix_len][num_partitions-1] + cost_of_last_substring)
    
    return dp[n][k]