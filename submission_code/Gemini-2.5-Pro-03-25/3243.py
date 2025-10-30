import sys
# Setting a higher recursion depth limit might be necessary for competitive programming platforms,
# although Python's default is often sufficient for typical constraints.
# sys.setrecursionlimit(2000) 

class Solution:
  """
  Solves the problem of counting powerful integers in a given range [start, finish].
  A positive integer x is called powerful if it ends with the suffix s (given as a string)
  and each digit in x is at most limit.
  The solution uses the principle of inclusion-exclusion combined with digit dynamic programming.
  The count for range [start, finish] is calculated as Count(finish) - Count(start - 1),
  where Count(N) is the number of powerful integers x such that 1 <= x <= N.
  """
  def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
      
      # Convert the suffix string s to an integer for an optimization check later.
      # This check is based on the fact that any number ending with suffix s must be at least int(s).
      s_val = int(s) 
      m = len(s) # Length of the suffix string s

      # memo dictionary will store results of DP states to avoid recomputation.
      # It will be reset within count_upto for each N.
      memo = {} 
      
      # N_str and n store the string representation of N and its length.
      # They are updated within count_upto for each N. Declared here to be accessible via nonlocal.
      N_str = "" 
      n = 0      

      # Helper function to compute count of powerful integers up to N
      def count_upto(N):
          """
          Computes the count of powerful integers x such that 1 <= x <= N.
          Uses digit dynamic programming (DP).
          """
          # Using nonlocal to indicate modification of variables from the nearest enclosing scope (numberOfPowerfulInt)
          # This allows resetting memo, N_str, n for each call to count_upto.
          nonlocal memo, N_str, n 
          
          # If N is negative, the count is 0. Base case for the subtraction logic.
          if N < 0: return 0 
          
          # Optimization: If N is less than the integer value of s, 
          # no positive integer x <= N can have s as a suffix, because the smallest such number is int(s).
          if N < s_val:
               return 0
              
          # Convert N to string to process its digits. Store its length.
          N_str = str(N)
          n = len(N_str)
          # Reset memoization cache for the current N value.
          memo = {} 
          
          # Define the core recursive DP function.
          # This nested function captures N_str, n, limit, s, m from its defining scope (count_upto).
          def dp(idx: int, tight: bool, is_leading_zero: bool):
              """
              Recursive DP function state.
              idx: current digit index being considered (from left, 0-based).
              tight: boolean, True if the number being built is currently restricted by the digits of N. 
                     False if it's already smaller than the prefix of N.
              is_leading_zero: boolean, True if we are currently placing leading zeros. 
                               False once the first non-zero digit is placed.
              Returns the count of valid ways to complete the number from this state.
              """
              
              # Base case: If we have processed all digits (reached the end of N_str length).
              if idx == n:
                  # If is_leading_zero is True, it means the number formed is 0 (all digits placed were 0).
                  # Since we are counting positive powerful integers, return 0.
                  # Otherwise, a valid positive number has been formed, return 1 to count it.
                  return 0 if is_leading_zero else 1

              # Memoization check: if this state has already been computed, return the stored result.
              state = (idx, tight, is_leading_zero)
              if state in memo:
                  return memo[state]

              res = 0
              # Determine the upper bound for the current digit.
              # If `tight` is True, the digit cannot exceed N_str[idx]. Otherwise, it can be up to 9.
              ub = int(N_str[idx]) if tight else 9

              # Iterate through all possible digits for the current position `idx`.
              for curr_digit in range(ub + 1):
                  
                  # Constraint: Current digit must not exceed the given `limit`.
                  if curr_digit > limit:
                      continue # Skip this digit if it violates the limit constraint.

                  # Constraint: If we are within the part of the number corresponding to the suffix s
                  # (i.e., the last m digits), the current digit must match the required digit from s.
                  # The suffix zone covers indices from n - m to n - 1.
                  if idx >= n - m:
                      # Calculate the index within the suffix string s corresponding to the current position idx.
                      s_idx = idx - (n - m)
                      target_digit = int(s[s_idx])
                      # Check if the current digit matches the target digit from s.
                      if curr_digit != target_digit:
                          continue # Skip this digit if it doesn't match the suffix requirement.

                  # Determine the tightness constraint for the next recursive call.
                  # If the current state is tight and we choose a digit equal to the upper bound,
                  # the next state remains tight. Otherwise, it becomes non-tight.
                  new_tight = tight and (curr_digit == ub)

                  if is_leading_zero:
                      # Handling the case where we are potentially placing leading zeros.
                      if curr_digit == 0:
                          # If placing 0 and still in leading zero state, recurse maintaining True state.
                          res += dp(idx + 1, new_tight, True)
                      else:
                          # Placing the first non-zero digit.
                          # Check if the resulting number's length is sufficient to contain the suffix s.
                          # The number's length starting from this first non-zero digit is n - idx.
                          current_len = n - idx 
                          if current_len < m:
                              # If the number length is less than suffix length m, this path is invalid
                              # because a number shorter than s cannot have s as suffix.
                              continue # Skip this digit.
                          
                          # Length is sufficient. Proceed recursively. Exit the leading zero state (set to False).
                          res += dp(idx + 1, new_tight, False)
                  else: 
                      # If not in leading zero state (already placed at least one non-zero digit).
                      # Continue building the number. The state remains is_leading_zero=False.
                      res += dp(idx + 1, new_tight, False)
              
              # Store the computed result for this state in the memoization table before returning.
              memo[state] = res
              return res

          # Start the DP computation from the most significant digit (index 0).
          # Initial state: tight=True (must not exceed N), is_leading_zero=True (start allows leading zeros).
          return dp(0, True, True)

      # Calculate the count of powerful integers up to `finish`.
      count_finish = count_upto(finish)
      # Calculate the count of powerful integers up to `start - 1`.
      count_start_minus_1 = count_upto(start - 1)
      
      # The final result is the difference, giving the count in the range [start, finish].
      return count_finish - count_start_minus_1