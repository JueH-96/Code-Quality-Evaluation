class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = 10**9

        # Precompute valid divisors for each length L (L >= 2)
        # For a given length L, we want all divisors d with 1 <= d < L such that L % d == 0.
        divisors = [[] for _ in range(n + 1)]
        for L in range(2, n + 1):
            for d in range(1, L):
                if L % d == 0:
                    divisors[L].append(d)

        # For speed, use s directly (string indexing in Python is fast enough).
        # Precompute the cost to transform any substring s[i:j+1] (with length >= 2)
        # into a semi-palindrome.
        # A semi-palindrome means there exists a divisor d (with 1 <= d < length and length % d == 0)
        # such that if we partition the substring into d groups based on indices mod d, each
        # resulting group (an arithmetic progression) can be turned into a palindrome.
        # For each group, the cost to make it a palindrome is the number of pairs of characters 
        # (from opposite ends) that differ (since one change per mismatched pair suffices).
        cost = [[0] * n for _ in range(n)]
        for i in range(n):
            # j must be at least i+1 (since substring length must be at least 2)
            for j in range(i + 1, n):
                L = j - i + 1
                best_cost = INF
                for d in divisors[L]:
                    group_count = L // d  # number of characters in each mod-class group
                    cur_cost = 0
                    # For each residue 0 <= r < d, take the arithmetic progression 
                    # positions: i + r, i + r + d, i + r + 2*d, ..., for group_count characters.
                    for r in range(d):
                        left, right = 0, group_count - 1
                        while left < right:
                            # Compare the characters in the group: if they differ, one change is needed.
                            if s[i + r + left * d] != s[i + r + right * d]:
                                cur_cost += 1
                            left += 1
                            right -= 1
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                cost[i][j] = best_cost

        # Next, we partition string s into exactly k contiguous substrings.
        # Let dp[i][p] be the minimum changes required to partition the prefix s[0:i]
        # into p substrings (each substring is required to have length at least 2).
        # The answer will be dp[n][k].
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # For each possible number of parts already formed (from 0 to k-1),
        # try to extend it by choosing a next substring s[i:j+1] (with length>=2).
        for parts in range(k):
            for i in range(n + 1):
                if dp[i][parts] < INF:
                    # j goes from i+1 to n-1 so that substring length = (j-i+1) is at least 2.
                    for j in range(i + 1, n):
                        # dp[j+1][parts+1] is updated with cost of adding substring s[i:j+1]
                        new_cost = dp[i][parts] + cost[i][j]
                        if new_cost < dp[j + 1][parts + 1]:
                            dp[j + 1][parts + 1] = new_cost
        return dp[n][k]