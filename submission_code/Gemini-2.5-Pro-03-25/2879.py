import math

class Solution:
  """
  Solves the problem of partitioning a string s into k substrings to minimize total changes 
  to make each substring a semi-palindrome. Uses dynamic programming.
  """
  def minimumChanges(self, s: str, k: int) -> int:
    """
    Calculates the minimum number of changes required to partition s into k semi-palindrome substrings.

    Args:
      s: The input string.
      k: The number of partitions required.

    Returns:
      An integer denoting the minimum total changes.
    """
    N = len(s)
    
    # Step 1: Precompute divisors for all possible substring lengths
    # all_divisors_list[L] will store all divisors d of L such that 1 <= d < L.
    all_divisors_list = [[] for _ in range(N + 1)]
    for L in range(2, N + 1): # Iterate over possible lengths L from 2 to N
        for d in range(1, int(L**0.5) + 1): # Check potential divisors up to sqrt(L)
            if L % d == 0:
                # d is a divisor
                if d < L: # Ensure divisor d is strictly less than L
                   all_divisors_list[L].append(d)
                # L // d is also a divisor
                d2 = L // d
                if d*d != L: # Avoid adding the same divisor twice if L is a perfect square
                   if d2 < L: # Ensure divisor d2 is strictly less than L
                       all_divisors_list[L].append(d2)

    # Step 2: Precompute the minimum changes cost for all possible substrings
    # min_changes_cost[i][j] stores the minimum changes needed for s[i...j] to become a semi-palindrome.
    min_changes_cost = [[0] * N for _ in range(N)] 

    for length in range(2, N + 1): # Iterate over possible substring lengths
        for i in range(N - length + 1): # Iterate over starting positions i
            j = i + length - 1 # Calculate ending position j
            L = length
            
            divisors = all_divisors_list[L] # Get the precomputed divisors for length L
            
            min_cost_for_substring = float('inf') # Initialize minimum cost for this substring
            
            # Calculate the cost for each valid divisor d and find the minimum
            for d in divisors:
                current_cost_for_d = 0 # Cost associated with this specific divisor d
                m = L // d # The length of each sequence formed by indices with the same modulo d
                
                # Iterate through the d sequences (one for each p from 0 to d-1)
                for p in range(d): 
                    # Check the palindrome property for the sequence corresponding to p
                    # The sequence indices are: i+p, i+p+d, ..., i+p+(m-1)*d
                    for k_idx in range(m // 2): # Compare elements symmetrically from start and end
                        idx1 = i + p + k_idx * d           # Index from the start of the sequence
                        idx2 = i + p + (m - 1 - k_idx) * d # Corresponding index from the end
                        if s[idx1] != s[idx2]:
                            current_cost_for_d += 1 # Increment cost if characters don't match
                
                # Update the minimum cost found so far for this substring
                min_cost_for_substring = min(min_cost_for_substring, current_cost_for_d)
            
            # Store the minimum cost for substring s[i...j]
            min_changes_cost[i][j] = min_cost_for_substring

    # Step 3: Dynamic Programming to find the minimum total cost for k partitions
    # dp[i][p_count] = minimum cost for partitioning the prefix s[0...i-1] into p_count substrings.
    dp = [[float('inf')] * (k + 1) for _ in range(N + 1)]
    dp[0][0] = 0 # Base case: cost of partitioning an empty string into 0 parts is 0.

    # Fill the DP table
    for p_count in range(1, k + 1): # Iterate through the number of partitions, from 1 to k
        for i_end in range(1, N + 1): # Iterate through the end index (exclusive) of the prefix s[0...i_end-1]
            
            # Determine the valid range for the start index (p_start) of the last (p_count-th) partition
            # The prefix s[0...p_start-1] must be long enough to accommodate p_count-1 partitions, each of minimum length 2.
            min_p_start = 2 * (p_count - 1) 
            # The last partition s[p_start...i_end-1] must have a minimum length of 2.
            max_p_start = i_end - 2 
            
            # If the range is invalid (min_p_start > max_p_start), it means s[0...i_end-1] cannot be partitioned into p_count valid substrings.
            # This typically happens when i_end is too small for the required number of partitions.
            if min_p_start > max_p_start:
                continue # Skip to the next i_end

            # Iterate through all possible start positions p_start for the last partition
            for p_start in range(min_p_start, max_p_start + 1): 
                 # Check if the state dp[p_start][p_count - 1] is reachable (i.e., has a finite cost)
                 # This state represents the minimum cost for partitioning the prefix s[0...p_start-1] into p_count-1 parts.
                 if dp[p_start][p_count - 1] != float('inf'):
                     # Retrieve the precomputed minimum changes cost for the last substring s[p_start...i_end-1]
                     last_substring_cost = min_changes_cost[p_start][i_end - 1]
                     
                     # Calculate the total cost if we use p_start as the beginning of the last partition
                     current_total_cost = dp[p_start][p_count - 1] + last_substring_cost
                     
                     # Update the DP state dp[i_end][p_count] with the minimum cost found so far for partitioning s[0...i_end-1] into p_count parts.
                     dp[i_end][p_count] = min(dp[i_end][p_count], current_total_cost)

    # The final result is stored in dp[N][k], representing the minimum cost for partitioning the entire string s (length N) into k parts.
    return dp[N][k]