import functools

class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)

        # Using functools.lru_cache for memoization.
        # The state is (i, j, remaining_k).
        @functools.lru_cache(None)
        def solve(i: int, j: int, rem_k: int) -> int:
            # Base case for recursion:
            # If pointers cross, the subsequence is empty (length 0).
            if i > j:
                return 0
            # If pointers meet, it's a single-character palindrome (length 1).
            if i == j:
                return 1

            # --- Recursive step ---

            # Option 1: Don't match s[i] and s[j] as the outer pair.
            # The longest palindromic subsequence is found by either excluding s[i]
            # or excluding s[j] from consideration for the endpoints.
            res = max(solve(i + 1, j, rem_k), solve(i, j - 1, rem_k))

            # Option 2: Match s[i] and s[j] to form the outer pair of the palindrome.
            # Calculate the minimum cost to make them the same character.
            diff = abs(ord(s[i]) - ord(s[j]))
            cost = min(diff, 26 - diff)
            
            # If we have enough budget (rem_k) for the transformation:
            if rem_k >= cost:
                # The length would be 2 (for the pair s[i], s[j]) plus the
                # length of the LPS in the inner substring s[i+1...j-1]
                # with the remaining budget.
                res = max(res, 2 + solve(i + 1, j - 1, rem_k - cost))
            
            return res

        return solve(0, n - 1, k)