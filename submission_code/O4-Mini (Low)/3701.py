class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        # If length < 3, impossible
        if n < 3:
            return ""
        # Precompute prefix sums PS[c][i]: cost to change caption[0..i-1] to char c
        # c index 0 for 'a', ..., 25 for 'z'
        PS = [[0] * (n + 1) for _ in range(26)]
        for ci in range(26):
            c_ord = ord('a') + ci
            ps = PS[ci]
            s = 0
            for i, ch in enumerate(caption):
                s += abs(ord(ch) - c_ord)
                ps[i+1] = s

        INF = 10**30
        # dp[i]: min cost to convert caption[:i] into good caption
        dp = [INF] * (n + 1)
        dp[0] = 0
        # best_c[ci]: min over j <= i-3 of (dp[j] - PS[ci][j])
        best_c = [INF] * 26
        # best_j[ci]: the j that attains best_c[ci]
        best_j = [-1] * 26
        # back pointers: for dp[i], we pick segment (j,i) with char c
        back = [(-1, -1)] * (n + 1)  # back[i] = (j, ci)

        for i in range(1, n + 1):
            # incorporate j = i-3 into best_c for future i
            j = i - 3
            if j >= 0:
                # for each character ci, update best_c[ci]
                for ci in range(26):
                    val = dp[j] - PS[ci][j]
                    if val < best_c[ci]:
                        best_c[ci] = val
                        best_j[ci] = j
            # now compute dp[i]
            # only possible if i >= 3, since best_c initialized only when j>=0
            if i >= 3:
                best_cost = INF
                pick_ci = -1
                pick_j = -1
                for ci in range(26):
                    cost = best_c[ci] + PS[ci][i]
                    if cost < best_cost:
                        best_cost = cost
                        pick_ci = ci
                        pick_j = best_j[ci]
                dp[i] = best_cost
                back[i] = (pick_j, pick_ci)

        # if impossible
        if dp[n] >= INF//2:
            return ""

        # Reconstruct segments
        res = []
        idx = n
        while idx > 0:
            j, ci = back[idx]
            # segment covers [j, idx), char = chr(ord('a')+ci)
            seg_len = idx - j
            res.append(chr(ord('a') + ci) * seg_len)
            idx = j
        res.reverse()
        return "".join(res)