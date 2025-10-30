import math # math is not strictly needed as float('inf') is available, but included for clarity or future use.

class Solution:
  """
  Solves the problem of finding the minimum number of beautiful substrings 
  a binary string can be partitioned into using dynamic programming.
  A substring is beautiful if it doesn't contain leading zeros and its
  integer value (from binary representation) is a power of 5.
  """
  def minimumBeautifulSubstrings(self, s: str) -> int:
      """
      Finds the minimum number of beautiful substrings partitioning s.

      Args:
        s: The input binary string.

      Returns:
        The minimum number of beautiful substrings, or -1 if impossible.
      """
      n = len(s)
      
      # Precompute the set of integer values that are powers of 5 and could 
      # potentially be represented by a substring of s.
      # The maximum length of s is 15 according to constraints.
      # The maximum value any substring can represent is 2^15 - 1 = 32767.
      # Any power of 5 greater than this value cannot be represented by a substring.
      beautiful_powers = set()
      power_of_5 = 1
      # Define the maximum value capacity based on the constraint N <= 15.
      # Any binary string of length k represents a value < 2^k. For k=15, value < 2^15.
      max_cap = (1 << 15) 
      
      while power_of_5 < max_cap: # Check against 2^15 limit
          beautiful_powers.add(power_of_5)
          # Calculate the next power of 5.
          # Check if the next power might exceed reasonable bounds or calculation capacity.
          # For Python's arbitrary precision integers, this primarily checks against max_cap.
          # If `power_of_5 * 5` would be >= max_cap, we stop. This is equivalent to
          # checking `power_of_5 >= max_cap / 5`. Integer division `//` is safer.
          if power_of_5 > max_cap // 5: 
               break
          power_of_5 *= 5
          # Note: The largest power of 5 less than 2^15 (32768) is 5^6 = 15625.
          # 5^7 = 78125 is too large. The loop correctly terminates after adding 15625.

      # Initialize DP table. dp[i] will store the minimum number of beautiful 
      # substrings needed to partition the prefix s[0...i-1].
      # Initialize with infinity, indicating that initially we assume it's impossible.
      dp = [float('inf')] * (n + 1)
      # Base case: The empty prefix s[0...-1] (represented by index 0) requires 0 substrings.
      dp[0] = 0  

      # Iterate through all possible prefix lengths i, from 1 up to n.
      # dp[i] corresponds to the prefix s[0...i-1].
      for i in range(1, n + 1):
          # For each prefix ending at index i-1, consider all possible substrings
          # that could end this prefix. A substring s[j...i-1] starts at index j.
          # The starting index j can range from 0 up to i-1.
          for j in range(i):
              # Extract the current substring candidate s[j...i-1].
              # In Python slicing, this corresponds to s[j:i].
              substring = s[j:i]
              
              # Check the conditions for this substring to be beautiful:
              
              # Condition 1: Must not have leading zeros. 
              # If the first character s[j] is '0', it's not beautiful.
              if substring[0] == '0':
                  continue # Skip this substring, try the next possible start index j.
              
              # Condition 2: Must be the binary representation of a power of 5.
              # Convert the binary substring to its integer value.
              val = int(substring, 2)
              
              # Check if this integer value is present in our precomputed set of powers of 5.
              if val in beautiful_powers:
                  # If the substring s[j:i] is indeed beautiful, we can potentially use it 
                  # as the last piece of a partition for the prefix s[0...i-1].
                  # This is only valid if the preceding part of the string, s[0...j-1], 
                  # can itself be partitioned into beautiful substrings.
                  # We check this condition by looking at dp[j].
                  if dp[j] != float('inf'):
                      # If dp[j] is finite, it means s[0...j-1] is partitionable.
                      # The total number of substrings for s[0...i-1] using this specific partition
                      # (ending with s[j:i]) would be dp[j] (for the prefix part) + 1 (for s[j:i]).
                      # We want the minimum count, so we update dp[i] with the minimum value found so far.
                      dp[i] = min(dp[i], dp[j] + 1)

      # After filling the DP table, dp[n] contains the minimum number of beautiful
      # substrings required to partition the entire string s = s[0...n-1].
      # If dp[n] is still infinity, it signifies that no valid partition exists.
      if dp[n] == float('inf'):
          return -1
      else:
          # Otherwise, a valid partition exists, and dp[n] holds the minimum count.
          # Return this minimum count. It's guaranteed to be an integer.
          return int(dp[n])