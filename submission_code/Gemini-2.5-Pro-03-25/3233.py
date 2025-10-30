import sys
# Setting a higher recursion depth might be necessary for deep recursive calls, 
# especially for larger N, though constraints might still lead to Time Limit Exceeded (TLE)
# If RecursionError occurs, an iterative DP approach would be needed.
# Use a reasonable limit like 2*N or N + buffer. Let's try N + a buffer.
try:
    # Set recursion depth slightly larger than N
    # sys.setrecursionlimit(10**4 + 50) 
    # Python environments might have varying limits or disallow setting it.
    # Removing explicit setting to avoid platform issues, rely on default or environment limits.
    pass
except Exception as e:
    # Optional: print a warning if setting fails, helps in debugging environment issues.
    # print(f"Warning: Could not set recursion depth: {e}")
    pass

class Solution:
  """
  Finds the maximum number of partitions of a string 's' after changing at most one character,
  where each partition is the longest possible prefix containing at most 'k' distinct characters.
  Uses Dynamic Programming with memoization.
  """
  def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
      n = len(s)
      
      # Memoization cache for the helper function get_end_optimized
      # Stores results for (start_idx, change_info) -> end_idx
      memo_get_end = {}

      # Helper function to find the end index of the first partition starting at start_idx.
      # It considers an optional character change specified by change_info=(p, c), where p is the index changed and c is the new character.
      # If change_info is None, it calculates based on the original string segment.
      # Time complexity: O(Length of partition * k) if using set, O(Length of partition) using dict/freq array. Max length is N.
      def get_end_optimized(start_idx, change_info):
          # Create a unique state key for memoization
          state = (start_idx, change_info)
          if state in memo_get_end:
              return memo_get_end[state]

          # Unpack change information
          p, char_c = change_info if change_info else (-1, '')

          # Use a dictionary for character frequency counting within the current partition window
          freq = {} 
          distinct_count = 0
          # Initialize the end index assuming an empty partition (-1 relative to start_idx)
          final_end_idx = start_idx - 1 

          # Iterate through the string starting from start_idx to find the partition boundary
          for curr_idx in range(start_idx, n):
              # Determine the character at curr_idx, considering the potential change
              char_to_process = s[curr_idx]
              if curr_idx == p:
                  char_to_process = char_c

              # Check the frequency of the character to process
              original_count = freq.get(char_to_process, 0)
              
              # If this character is new to the current partition window
              if original_count == 0: 
                  # Check if adding this new distinct character would exceed the limit k
                  if distinct_count == k:
                      # If limit k is reached, the partition cannot include this character. Break the loop.
                      break 
                  # Otherwise, increment the distinct character count
                  distinct_count += 1 

              # Update the frequency count for the character
              freq[char_to_process] = original_count + 1
              # Since the character is included, update the potential end index of the partition
              final_end_idx = curr_idx 
          
          # Memoize the calculated end index for this state
          memo_get_end[state] = final_end_idx
          return final_end_idx

      # Memoization cache for the main DP function
      # Stores results for (idx, change_used) -> max_partitions
      memo_dp = {}

      # DP function: dp(idx, change_used) returns the maximum number of partitions
      # for the suffix of the string starting at index `idx`.
      # `change_used` is a boolean indicating if the one allowed change has already been utilized.
      def dp(idx, change_used):
          # Create a unique state key for memoization
          state = (idx, change_used)
          # Base case: If we've processed the entire string, no more partitions can be formed.
          if idx >= n: 
              return 0
          # Return memoized result if available
          if state in memo_dp: 
              return memo_dp[state]

          res = 0
          if change_used:
              # If the single allowed change has already been used:
              # We must proceed without making any further changes.
              # Find the end `j` of the current partition starting at `idx` using the original string segment.
              j = get_end_optimized(idx, None) 
              # The total partitions = 1 (for the current partition) + partitions from the rest of the string.
              res = 1 + dp(j + 1, True) # change_used remains True
          else:
              # If the change is still available:
              # We have two choices: either use the change now or save it for later.

              # Option 1: Don't use the change in the current partition.
              # Find the end `j_nochange` of the partition starting at `idx` using original string segment.
              j_nochange = get_end_optimized(idx, None)
              # The change remains available (change_used = False) for the recursive call.
              res_option1 = 1 + dp(j_nochange + 1, False)

              # Option 2: Use the change in the current partition (or potentially later if optimal).
              # Initialize the maximum result for this option.
              max_res_option2 = 0 
              
              # Iterate through all possible positions `p` from `idx` onwards to make the change.
              for p in range(idx, n): 
                  original_char = s[p]
                  # Iterate through all possible characters `c` ('a' through 'z') to change to.
                  for char_code in range(ord('a'), ord('z') + 1):
                      c = chr(char_code)
                      # Skip if the character `c` is the same as the original character at `p`.
                      if c == original_char: continue
                      
                      # Calculate the end index `j_prime` of the partition starting at `idx`
                      # assuming the character at index `p` is changed to `c`.
                      j_prime = get_end_optimized(idx, (p, c))
                      
                      # Check if the change position `p` falls within the bounds [idx, j_prime] of the resulting partition.
                      # If `p > j_prime`, the change occurs after this partition ends, so it doesn't count as using the change "now".
                      # This scenario is implicitly covered under Option 1 where the change is deferred.
                      if p <= j_prime: 
                         # If the change is made within this partition:
                         # Calculate the total number of partitions. Add 1 for the current partition.
                         # The recursive call uses `change_used = True` because the change has been consumed.
                         current_res_option2 = 1 + dp(j_prime + 1, True) 
                         # Update the maximum partitions found so far using Option 2.
                         max_res_option2 = max(max_res_option2, current_res_option2)
              
              # The final result for `dp(idx, False)` is the maximum of Option 1 and Option 2.
              res = max(res_option1, max_res_option2)
          
          # Memoize the result for the current state before returning.
          memo_dp[state] = res
          return res

      # Initial call to the DP function starts at index 0, with the change available (False).
      return dp(0, False)