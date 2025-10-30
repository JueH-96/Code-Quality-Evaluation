from array import array

class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        # If length < 3, cannot form any good caption
        if n < 3:
            return ""
        # Convert caption to integer list 0..25
        base_a = ord('a')
        s = [ord(ch) - base_a for ch in caption]
        # Infinity large cost
        INF = 10**9
        # We keep DP states DP_prev[c*3 + l] for l=0,1,2 representing run lengths 1,2,>=3
        # And pointers in a flat array of size n * (26*3) = n*78
        PTR = array('I', [0]) * (n * 78)
        # Initialize DP_prev for position 0
        DP_prev = [INF] * (26*3)
        c0 = s[0]
        # At i=0, run length must be 1 for whichever c we pick
        for c in range(26):
            DP_prev[c*3 + 0] = abs(c0 - c)
            # DP_prev[c*3 + 1], DP_prev[c*3 + 2] remain INF

        # Forward DP, store pointers for backtracking
        for i in range(1, n):
            si = s[i]
            # Precompute two smallest DP_prev[*][2] and their char indexes
            best1_val = INF
            best2_val = INF
            best1_char = -1
            best2_char = -1
            prev_base = 0
            # l_index = 2 means run length >= 3 in DP_prev
            for c in range(26):
                v = DP_prev[c*3 + 2]
                if v < best1_val:
                    best2_val, best2_char = best1_val, best1_char
                    best1_val, best1_char = v, c
                elif v < best2_val:
                    best2_val, best2_char = v, c
            # Prepare DP_curr
            DP_curr = [INF] * (26*3)
            row_base = i * 78
            # For each possible char c at pos i
            for c in range(26):
                cost = abs(si - c)
                idx0 = c*3 + 0  # new run length = 1
                # break from previous run, only if we had run>=3 of a different char
                if best1_char != -1:
                    if c != best1_char:
                        v0 = cost + best1_val
                        if v0 < DP_curr[idx0]:
                            DP_curr[idx0] = v0
                            # pointer to prev state (best1_char, l=2)
                            PTR[row_base + idx0] = (best1_char << 2) | 2
                    else:
                        # must use second best if prev_c == c
                        if best2_char != -1:
                            v0 = cost + best2_val
                            if v0 < DP_curr[idx0]:
                                DP_curr[idx0] = v0
                                PTR[row_base + idx0] = (best2_char << 2) | 2
                # extend run from length 1 -> 2
                idx1 = c*3 + 1
                v1 = cost + DP_prev[c*3 + 0]
                if v1 < DP_curr[idx1]:
                    DP_curr[idx1] = v1
                    PTR[row_base + idx1] = (c << 2) | 0
                # extend run from length 2->3 or keep >=3
                idx2 = c*3 + 2
                prev1 = DP_prev[c*3 + 1]
                prev2 = DP_prev[c*3 + 2]
                if prev1 <= prev2:
                    v2 = cost + prev1
                    prev_l = 1
                else:
                    v2 = cost + prev2
                    prev_l = 2
                if v2 < DP_curr[idx2]:
                    DP_curr[idx2] = v2
                    PTR[row_base + idx2] = (c << 2) | prev_l
            # Move DP_curr to DP_prev
            DP_prev = DP_curr

        # At the last position, only states with run length>=3 are valid
        # Find minimal cost and lexicographically smallest char for that cost
        best_cost = INF
        start_c = -1
        for c in range(26):
            v = DP_prev[c*3 + 2]
            if v < best_cost:
                best_cost = v
                start_c = c
        if best_cost >= INF:
            return ""  # impossible

        # Backtrack to build the result string
        res = [''] * n
        c = start_c
        l_index = 2
        # from i=n-1 downto 1, follow pointers
        for i in range(n-1, 0, -1):
            res[i] = chr(base_a + c)
            code = PTR[i*78 + (c*3 + l_index)]
            prev_l = code & 3
            prev_c = code >> 2
            c, l_index = prev_c, prev_l
        # finally position 0
        res[0] = chr(base_a + c)
        return "".join(res)