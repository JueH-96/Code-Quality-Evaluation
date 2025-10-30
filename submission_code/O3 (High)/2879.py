class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = 10 ** 9

        # ------------------------------------------------------------
        # 1.  pre–compute all proper divisors   1 <= d < L ,  L % d == 0
        # ------------------------------------------------------------
        divisors = [[] for _ in range(n + 1)]
        for L in range(2, n + 1):
            for d in range(1, L):
                if L % d == 0:
                    divisors[L].append(d)          # d is a proper divisor

        # ------------------------------------------------------------
        # 2.  costSemi[l][r]  :  minimum number of changes needed to
        #                       turn s[l..r] (len>=2) into a semi-palindrome
        # ------------------------------------------------------------
        costSemi = [[INF] * n for _ in range(n)]

        for l in range(n):
            for r in range(l + 1, n):              # length at least 2
                L = r - l + 1
                best = INF
                for d in divisors[L]:
                    m = L // d                     # length of each mod-class subsequence
                    cost = 0
                    for c in range(d):             # for every residue class
                        for t in range(m // 2):    # compare symmetric positions
                            i = l + c + t * d
                            j = l + c + (m - 1 - t) * d
                            if s[i] != s[j]:
                                cost += 1
                    if cost < best:
                        best = cost
                        if best == 0:              # cannot do better than 0
                            break
                costSemi[l][r] = best

        # ------------------------------------------------------------
        # 3.  DP over partitions
        #     dp[pos][parts] = min cost for first 'pos' characters
        #                      split into 'parts' semi-palindrome pieces
        # ------------------------------------------------------------
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for end in range(1, n + 1):                      # prefix length
            max_parts_here = min(k, end // 2)            # need at least 2 chars/part
            for parts in range(1, max_parts_here + 1):
                # the first (parts-1) pieces need at least 2*(parts-1) characters
                start_min = 2 * (parts - 1)
                for start in range(start_min, end - 1):  # segment length ≥ 2
                    if dp[start][parts - 1] == INF:
                        continue
                    seg_cost = costSemi[start][end - 1]
                    new_cost = dp[start][parts - 1] + seg_cost
                    if new_cost < dp[end][parts]:
                        dp[end][parts] = new_cost

        return dp[n][k]