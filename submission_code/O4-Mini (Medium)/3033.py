class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        # collect mismatch positions
        pos = [i for i in range(n) if s1[i] != s2[i]]
        m = len(pos)
        # if odd number of mismatches, impossible
        if m & 1:
            return -1
        # no mismatches
        if m == 0:
            return 0
        INF = 10**18
        # f[l][r] = min cost to perfectly match all mismatches in pos[l..r]
        # where (r - l + 1) is even
        f = [[INF] * m for _ in range(m)]
        # base case: segments of length 2
        for i in range(m - 1):
            dist = pos[i+1] - pos[i]
            # either use global flip cost x, or adjacent-chain cost = distance
            f[i][i+1] = dist if dist < x else x
        
        # build up for even lengths >= 4
        for length in range(4, m + 1, 2):
            # l..r inclusive is our segment
            for l in range(0, m - length + 1):
                r = l + length - 1
                # option 1: pair l with r, and match the inner segment
                lr_cost = pos[r] - pos[l]
                if lr_cost > x:
                    lr_cost = x
                best = lr_cost + f[l+1][r-1]
                # option 2: split the segment into two even parts
                # split point k where [l..k] and [k+1..r] both even-length
                # that requires (k - l + 1) even => k - l is odd => step by 2 from l+1
                for k in range(l+1, r, 2):
                    cost_split = f[l][k] + f[k+1][r]
                    if cost_split < best:
                        best = cost_split
                f[l][r] = best
        
        # answer is cost to match all mismatches
        return f[0][m-1]