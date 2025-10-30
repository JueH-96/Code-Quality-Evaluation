import sys

class Solution:
  """
  Finds the minimum number of operations (digit deletions) to make the number represented by the string 'num' divisible by 25.
  A number is divisible by 25 if its last two digits form "00", "25", "50", or "75".
  Additionally, 0 is divisible by 25. Since deleting all digits results in 0, this is always possible.
  """
  def minimumOperations(self, num: str) -> int:
      """
      Calculates the minimum operations needed by checking possibilities.

      Args:
          num: The input string representing a non-negative integer.

      Returns:
          The minimum number of deletions required to make the number special (divisible by 25).
      """
      n = len(num)
      # Initialize min_ops. The maximum possible operations is n, 
      # which corresponds to deleting all digits to get 0.
      # This serves as an initial upper bound and covers the case where '0' is not present 
      # and no special suffix can be formed.
      min_ops = n  

      # Check if '0' exists in the number string.
      # If '0' exists, we have an option to delete all digits except one '0'.
      # This results in the number 0, which is divisible by 25.
      # The number of operations for this is n - 1 (delete all but one character).
      # We update min_ops with this possibility if '0' is present.
      # Note: If num is "0", n=1, then n-1=0 deletions.
      if '0' in num:
          min_ops = n - 1 
      
      # Now, search for possibilities to form a number ending in "00", "25", "50", or "75".
      # These are the required two-digit suffixes for divisibility by 25 (excluding 0 itself).
      # For each target suffix XY:
      # 1. Find the rightmost occurrence of the units digit Y at index j.
      # 2. If Y is found, find the rightmost occurrence of the tens digit X at index i such that i < j.
      # 3. If both X and Y are found in the correct order (i < j), calculate the number of deletions.
      # The number of deletions is the count of digits between i and j plus the count of digits after j.
      # Deletions = (idx_j - idx_i - 1) + (n - 1 - idx_j) = n - idx_i - 2.
      # We update min_ops with the minimum deletions found across all target suffixes.
      
      targets = ["00", "25", "50", "75"]
      
      for target in targets:
          target_digit_1 = target[0] # The required tens digit
          target_digit_2 = target[1] # The required units digit
          
          # Find the rightmost index `idx_j` for `target_digit_2` (units digit).
          # The `rfind` method returns the highest index in the string where the substring is found,
          # or -1 if not found.
          idx_j = num.rfind(target_digit_2)

          # If the units digit is found (index is not -1)
          if idx_j != -1:
              # Find the rightmost index `idx_i` for `target_digit_1` (tens digit) 
              # such that `idx_i < idx_j`. 
              # We search in the substring `num[0...idx_j-1]` using `rfind` with slice bounds.
              # The slice end index `idx_j` is exclusive in `rfind(sub, start, end)`.
              # This ensures we find an index `i` strictly less than `j`.
              idx_i = num.rfind(target_digit_1, 0, idx_j)

              # If the tens digit is also found at a valid index idx_i before idx_j
              if idx_i != -1:
                  # Calculate the number of deletions required for this pair (idx_i, idx_j).
                  # The digits removed are those strictly between index i and j, and those strictly after index j.
                  # Number of digits between i and j = idx_j - idx_i - 1
                  # Number of digits after j = n - (idx_j + 1) = n - 1 - idx_j
                  # Total deletions = (idx_j - idx_i - 1) + (n - 1 - idx_j) 
                  # Simplifying this expression gives: n - idx_i - 2
                  deletions = n - idx_i - 2
                  
                  # Update the minimum operations found so far if this path requires fewer deletions.
                  min_ops = min(min_ops, deletions)

      # Return the overall minimum operations found after checking the '0' case and all target suffixes.
      return min_ops