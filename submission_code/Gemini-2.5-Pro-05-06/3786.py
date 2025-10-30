import functools

class Solution:
  def longestPalindromicSubsequence(self, s: str, k: int) -> int:
    n = len(s)
    
    # Precompute character values (0-25) for quick lookup
    # s_val[i] stores ord(s[i]) - ord('a')
    s_val = [ord(c) - ord('a') for c in s]

    # Helper function to calculate character difference cost.
    # It's defined here to be part of the class or can be a static/local helper.
    # It doesn't need memoization as it's a simple calculation.
    def calculate_char_diff_static(char_val1: int, char_val2: int) -> int:
        diff = abs(char_val1 - char_val2)
        return min(diff, 26 - diff)

    # dp(i, j, k_rem) is the length of the longest palindromic subsequence
    # in s[i...j] using at most k_rem operations.
    # Using functools.lru_cache for memoization.
    @functools.lru_cache(None)
    def dp(i: int, j: int, k_rem: int) -> int:
      # Base case 1: If remaining operations are negative, this path is invalid.
      # Return a very small number (negative infinity) to ensure it's not chosen by max().
      if k_rem < 0:
        return -float('inf') 
      
      # Base case 2: If pointers cross (i > j), the substring is empty. LPS length is 0.
      if i > j:
        return 0
      
      # Base case 3: If substring is a single character (i == j). LPS length is 1.
      # This costs 0 operations, as the character itself forms a palindrome.
      # This is valid as long as k_rem >= 0 (which is true if this point is reached).
      if i == j:
        return 1
      
      # Recursive step:
      
      # Option 1: s[i] and s[j] form the ends of the palindrome.
      # They need to be made equal. Cost is diff(s[i], s[j]).
      cost_to_match_outer_pair = calculate_char_diff_static(s_val[i], s_val[j])
      # The length from this option would be 2 (for s[i] and s[j])
      # plus the LPS of the inner substring s[i+1...j-1].
      # The operations budget for the inner part is k_rem - cost_to_match_outer_pair.
      res_option1 = 2 + dp(i + 1, j - 1, k_rem - cost_to_match_outer_pair)
      
      # Option 2: s[i] is not part of the palindrome (i.e., skip s[i] for the outer pair).
      # The LPS is then sought in s[i+1...j] with the same k_rem budget.
      res_option2 = dp(i + 1, j, k_rem)
      
      # Option 3: s[j] is not part of the palindrome (i.e., skip s[j] for the outer pair).
      # The LPS is then sought in s[i...j-1] with the same k_rem budget.
      res_option3 = dp(i, j - 1, k_rem)
      
      # The result for dp(i, j, k_rem) is the maximum of these options.
      result = max(res_option1, res_option2, res_option3)
      return result

    # Initial call for the entire string s (indices 0 to n-1) with the full budget k.
    # The lru_cache is associated with the `dp` function. If `Solution` instances
    # are created per test (common on platforms like LeetCode), the cache is fresh.
    # If the same instance were used for multiple `longestPalindromicSubsequence` calls
    # with potentially different `s` or `s_val`, `dp.cache_clear()` would be needed.
    # Here, `dp` is defined within the method, so it's fine; it captures the correct `s_val`.
    
    final_ans = dp(0, n - 1, k)
    
    # Since n >= 1 (from constraints) and k >= 1 (from constraints, though k=0 also works),
    # a palindrome of length at least 1 (a single character) is always possible with 0 cost.
    # So, final_ans will be >= 1. No need to handle -float('inf') specifically for the final result.
    return final_ans