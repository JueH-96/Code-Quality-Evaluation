def cost(c1: str, c2: str) -> int:
    """Calculates the minimum operations to change c1 to c2 wrapping around alphabet."""
    p1 = ord(c1) - ord('a')
    p2 = ord(c2) - ord('a')
    diff = abs(p1 - p2)
    return min(diff, 26 - diff)

class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        # dp[i][j][c] = max length of palindromic subsequence of s[i...j] using at most c operations on the subsequence characters
        # i: start index of substring (0 to n-1)
        # j: end index of substring (0 to n-1)
        # c: maximum cost allowed for the subsequence from s[i...j] (0 to k)
        # Initialize with 0s. dp table size n x n x (k+1).
        dp = [[[0 for _ in range(k + 1)] for _ in range(n)] for _ in range(n)]

        # Iterate by length of substring (len = j - i + 1)
        # Smallest length is 1, largest length is n.
        for length in range(1, n + 1):
            # Iterate by starting index i
            # For a given length, i can range from 0 up to n - length.
            for i in range(n - length + 1):
                # Calculate the corresponding ending index j
                j = i + length - 1

                # Iterate through all possible cost budgets c from 0 up to k
                for c in range(k + 1):
                    if length == 1:
                        # Base case: substring s[i...i] contains a single character.
                        # A single character forms a palindromic subsequence of length 1.
                        # This subsequence requires 0 operations to be a palindrome (it is already).
                        # The cost budget 'c' doesn't affect the length of this single-character subsequence.
                        dp[i][j][c] = 1
                    else: # length > 1 (i < j)
                        # Recurrence relation for substrings of length greater than 1.
                        # We consider three possibilities for forming the longest palindromic subsequence of s[i...j]:

                        # Option 1: The LPS of s[i...j] does *not* include the character s[j].
                        # In this case, the LPS must be formed from the substring s[i...j-1].
                        # The cost budget c remains the same for the subproblem s[i...j-1].
                        dp[i][j][c] = dp[i][j-1][c]

                        # Option 2: The LPS of s[i...j] does *not* include the character s[i].
                        # In this case, the LPS must be formed from the substring s[i+1...j].
                        # We take the maximum length seen so far (from Option 1) and compare it with this option.
                        # The cost budget c remains the same for the subproblem s[i+1...j].
                        dp[i][j][c] = max(dp[i][j][c], dp[i+1][j][c])

                        # Option 3: The LPS of s[i...j] includes *both* s[i] and s[j] as the outermost characters of the palindrome.
                        # To form a palindrome, the characters corresponding to s[i] and s[j] in the subsequence must be made equal.
                        # The minimum number of operations required to make s[i] and s[j] match is given by the cost function.
                        cost_to_match = cost(s[i], s[j])

                        # If the current cost budget 'c' is large enough to afford the operations needed to match s[i] and s[j]:
                        if c >= cost_to_match:
                            # We use `cost_to_match` operations on s[i] and s[j].
                            # The remaining part of the palindromic subsequence must be an LPS of the inner substring s[i+1...j-1].
                            # The operations budget remaining for the inner subproblem is c - cost_to_match.

                            inner_len = 0 # Initialize the length of the LPS from the inner substring to 0.
                            # Check if the inner substring s[i+1...j-1] is a valid non-empty range.
                            # This occurs when i + 1 <= j - 1, which is equivalent to i + 2 <= j, or length >= 3.
                            if i + 1 <= j - 1:
                                # If the inner substring is valid and non-empty, we look up the max LPS length
                                # for this substring with the reduced cost budget.
                                inner_len = dp[i+1][j-1][c - cost_to_match]

                            # The total length of the palindromic subsequence in this option is 2 (for the matched s[i] and s[j])
                            # plus the maximum length of the LPS found from the inner substring.
                            # We take the maximum among all considered options.
                            dp[i][j][c] = max(dp[i][j][c], 2 + inner_len)

        # The final answer is the maximum length of a palindromic subsequence for the entire string s[0...n-1]
        # using a total cost of at most k operations, which is stored in dp[0][n-1][k].
        return dp[0][n-1][k]