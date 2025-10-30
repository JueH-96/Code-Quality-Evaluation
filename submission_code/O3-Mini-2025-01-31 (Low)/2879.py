class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = float('inf')
        # Precompute cost[i][j] = minimum changes to turn s[i:j+1] into a semi-palindrome, or INF if impossible.
        # A substring must have length >= 2 to be semi-palindrome (since d must be < len and >= 1)
        cost = [[INF] * n for _ in range(n)]
        
        # Helper: Given a list of characters, compute the minimum changes to make it palindrome.
        # Standard pairwise matching differences.
        def palindromeCost(chars):
            L = len(chars)
            c = 0
            i, j = 0, L - 1
            while i < j:
                if chars[i] != chars[j]:
                    c += 1
                i += 1
                j -= 1
            return c

        # For every substring s[i:j+1], try every valid divisor d and record minimal cost.
        # Note: We need d with 1 <= d < L and L % d == 0.
        for i in range(n):
            # j from i to end gives substring length L = j - i + 1.
            for j in range(i, n):
                L = j - i + 1
                # length 1 cannot be semi-palindrome.
                if L < 2:
                    continue
                best = INF
                # iterate over possible divisors d such that 1<= d < L and L % d == 0.
                for d in range(1, L):
                    if L % d != 0:
                        continue
                    total = 0
                    # For each modulo group 0...d-1, gather group elements.
                    for r in range(d):
                        group = []
                        idx = i + r
                        while idx <= j:
                            group.append(s[idx])
                            idx += d
                        # cost to convert this group to a palindrome:
                        total += palindromeCost(group)
                    best = min(best, total)
                cost[i][j] = best
        
        # Now partition s into k contiguous substrings minimizing sum of cost
        # Let dp[i][p] = minimum cost to partition s[0:i] (first i characters) into p parts.
        dp = [[INF]*(k+1) for _ in range(n+1)]
        dp[0][0] = 0
        
        for i in range(1, n+1):
            for p in range(1, k+1):
                # Try every possible previous cut position.
                for j in range(0, i):
                    if dp[j][p-1] == INF:
                        continue
                    # substring from j to i-1
                    if cost[j][i-1] == INF:
                        continue
                    dp[i][p] = min(dp[i][p], dp[j][p-1] + cost[j][i-1])
        return dp[n][k]
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    s = "abcac"
    k = 2
    print(sol.minimumChanges(s, k))  # Expected output: 1

    # Example 2:
    s = "abcdef"
    k = 2
    print(sol.minimumChanges(s, k))  # Expected output: 2

    # Example 3:
    s = "aabbaa"
    k = 3
    print(sol.minimumChanges(s, k))  # Expected output: 0