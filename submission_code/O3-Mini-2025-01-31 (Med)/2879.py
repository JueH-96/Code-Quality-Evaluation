class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = 10**9

        # Precompute a dictionary of valid divisors for lengths from 2 to n.
        # For a given length L, valid divisors are those d with 1 <= d < L and L % d == 0.
        valid_divs = {}
        for L in range(2, n+1):
            valid_divs[L] = []
            for d in range(1, L):
                if L % d == 0:
                    valid_divs[L].append(d)
                    
        # Precompute cost[i][j] = minimal changes required to convert s[i:j+1] into a semi-palindrome.
        # If s[i:j+1] is length L, we try each valid d in valid_divs[L]:
        # For each residue r in 0..d-1, consider the group G = s[i+r], s[i+r+d], ..., s[j].
        # To make G a palindrome, we can change letters at pairs (l, m) (two pointers).
        # Sum the changes over groups, and take minimum over d.
        cost = [[INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):  # substring length at least 2.
                L = j - i + 1
                best = INF
                # Try each possible d that divides L (with 1 <= d < L)
                for d in valid_divs[L]:
                    cur_cost = 0
                    # For each residue mod d, create the group from indices i+r, i+r+d, ...
                    for r in range(d):
                        # Build the group indices: they are i + r, i + r + d, ..., <= j.
                        # Instead of creating a list, we can compute cost pair-wise.
                        # We compute indices in group: they are i+r, i+r+d, ..., up to j.
                        group = []
                        idx = i + r
                        while idx <= j:
                            group.append(s[idx])
                            idx += d
                        # Now compute minimal changes to make group a palindrome.
                        m = len(group)
                        group_cost = 0
                        left, right = 0, m - 1
                        while left < right:
                            if group[left] != group[right]:
                                group_cost += 1
                            left += 1
                            right -= 1
                        cur_cost += group_cost
                    best = min(best, cur_cost)
                cost[i][j] = best

        # Now we need to partition s into k contiguous substrings,
        # each substring (segment) must be convertible into a semi-palindrome.
        # Note: each substring must have length at least 2 (otherwise it's impossible).
        # DP: dp[i][seg] = minimum cost to partition s[0:i] into seg segments.
        # We want dp[n][k].
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # Because every segment must have length >= 2, dp[i][seg] is only feasible if i >= 2*seg.
        for i in range(2, n + 1):
            # seg segments, seg can be at most min(k, i//2)
            for seg in range(1, min(k, i // 2) + 1):
                # We want to choose a previous cut index j, where substring s[j:i] (i.e. indices j..i-1) is valid.
                # s[j:i] must have length at least 2, i.e., j <= i - 2.
                for j in range(2 * (seg - 1), i - 1):  # ensures previous seg-1 segments can cover at least 2 chars each.
                    # Only if dp[j][seg-1] is feasible.
                    if dp[j][seg - 1] < INF:
                        # substring from j to i-1 has cost cost[j][i-1]
                        cur_seg_cost = cost[j][i - 1]
                        if cur_seg_cost < INF:
                            dp[i][seg] = min(dp[i][seg], dp[j][seg - 1] + cur_seg_cost)

        return dp[n][k]