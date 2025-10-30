import math

class Solution:
  """
  This class provides a solution to find the longest palindromic subsequence
  of a given string 's' with at most 'k' character modifications allowed.
  Each modification allows changing a character to its adjacent character in the
  alphabet (wrapping around 'a' and 'z').
  """
  def longestPalindromicSubsequence(self, s: str, k: int) -> int:
    """
    Calculates the length of the longest palindromic subsequence.

    Args:
      s: The input string containing lowercase English letters.
      k: The maximum number of allowed character modifications.

    Returns:
      The maximum length of a palindromic subsequence achievable within k modifications.
    """
    
    n = len(s)

    # Helper function to calculate the minimum distance (cost) between two characters
    # on the circular alphabet ('a'...'z'). The cost represents the minimum number
    # of operations needed to change one character to another.
    def dist(c1, c2):
        """Calculates the minimum operations to change c1 to c2."""
        ord1 = ord(c1) - ord('a')
        ord2 = ord(c2) - ord('a')
        diff = abs(ord1 - ord2)
        # The distance is the minimum of the direct path and the wrap-around path.
        return min(diff, 26 - diff)

    # Base case: If the string has only one character, it's a palindrome of length 1.
    # No modifications needed (cost 0).
    if n == 1:
        return 1

    # Dynamic Programming Approach with Space Optimization: O(N*K) space complexity.
    # The standard DP approach for Longest Palindromic Subsequence (LPS) has state dp[i][j] 
    # representing the LPS length for substring s[i...j].
    # Here, we need to incorporate the cost 'k'. So the state becomes dp[i][j][c],
    # representing the max LPS length for s[i...j] using at most 'c' cost.
    # The state space is O(N^2 * K).
    # We can optimize space to O(N*K) because computing dp for length L only depends
    # on results for lengths L-1 and L-2. We use three DP tables:
    # `curr` for current length L, `prev` for length L-1, `prev2` for length L-2.

    # Initialize DP table for length L=1 (base case).
    # `prev2[i][c]` stores the result for substring s[i...i] (length 1).
    # Any single character is a palindrome of length 1 with cost 0.
    # So, `dp[i][i][c] = 1` for all `c >= 0`.
    prev2 = [[1] * (k + 1) for _ in range(n)] 
    
    # Initialize DP table for length L=2.
    # `prev[i][c]` stores the result for substring s[i...i+1] (length 2).
    prev = [[0] * (k + 1) for _ in range(n)]
    for i in range(n - 1):
        # Calculate the minimum cost to make s[i] and s[i+1] identical.
        # This is the cost required if we want to form a palindrome of length 2 using these two characters.
        cost_match = dist(s[i], s[i+1]) 
        for c in range(k + 1): # Iterate through possible budgets from 0 to k.
            # The maximum length considering just one character from s[i], s[i+1] is 1.
            # This corresponds to max(dp[i+1][i+1][c], dp[i][i][c]) = max(prev2[i+1][c], prev2[i][c]) = max(1, 1) = 1.
            prev[i][c] = 1
            
            # Check if we have enough budget 'c' to match s[i] and s[i+1].
            if c >= cost_match:
                # If matched, the length becomes 2 (by forming the pair s[i], s[i+1]).
                # The cost incurred is `cost_match`.
                prev[i][c] = max(prev[i][c], 2)

    # Compute DP states for lengths L = 3 to n.
    for L in range(3, n + 1):
        # `curr` table stores results for the current length L.
        curr = [[0] * (k + 1) for _ in range(n)]
        # Iterate through all possible starting positions 'i' for substrings of length L.
        for i in range(n - L + 1):
            j = i + L - 1  # Ending position of the substring s[i...j].
            # Calculate the minimum cost to make the outer characters s[i] and s[j] match.
            cost_match = dist(s[i], s[j]) 

            for c in range(k + 1): # Iterate through budget 'c' from 0 to k.
                # Calculate the current state `curr[i][c]` based on previously computed states.
                
                # Option 1 & 2: Don't match s[i] and s[j].
                # The length is the maximum of the LPS lengths of the two subproblems of length L-1:
                # - Subproblem s[i+1...j] (LPS length stored in `prev[i+1][c]`)
                # - Subproblem s[i...j-1] (LPS length stored in `prev[i][c]`)
                curr[i][c] = max(prev[i+1][c], prev[i][c])
                
                # Option 3: Match s[i] and s[j].
                # This is possible only if the current budget 'c' is sufficient (c >= cost_match).
                if c >= cost_match:
                    # If matched, the length increases by 2 (for the pair s[i], s[j]).
                    # The total length is 2 plus the LPS length of the inner substring s[i+1...j-1].
                    # The inner substring has length L-2. Its results are stored in `prev2`.
                    # We need the result for the inner substring starting at `i+1`, 
                    # with the remaining budget `c - cost_match`. This is `prev2[i+1][c - cost_match]`.
                    curr[i][c] = max(curr[i][c], 2 + prev2[i+1][c - cost_match])
        
        # Update DP tables for the next iteration: Shift the tables.
        # `prev` becomes `prev2`, and `curr` becomes `prev`.
        prev2 = prev
        prev = curr

    # After the loops complete, `prev[0][k]` contains the maximum LPS length 
    # for the entire string s[0...n-1] using at most budget `k`.
    # This logic correctly handles the case n=2 as well, because the loop `range(3, n+1)`
    # will be empty (`range(3, 3)`), and `prev` will hold the results for L=2.
    return prev[0][k]