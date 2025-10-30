class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        # collect positions of '0'
        zeros = [i for i, ch in enumerate(s) if ch == '0']
        m = len(zeros)
        # sentinel array of zero‐positions: -1 at front, n at end
        pl = [-1] + zeros + [n]
        
        ans = 0
        # z = 0 case: substrings with no zeros are just runs of ones
        prev = pl[0]
        for curr in pl[1:]:
            L = curr - prev - 1
            if L > 0:
                ans += L * (L + 1) // 2
            prev = curr
        
        # z >= 1 case
        # for a substring with exactly z zeros, let K = z^2 + z.
        # we need length >= K to satisfy ones >= z^2.
        for z in range(1, n + 1):
            K = z * (z + 1)
            if K > n:
                break
            # number of ways to pick a block of z zeros in the zero‐positions array
            end = m - z + 1
            if end <= 0:
                continue
            # slide over each group of z consecutive zeros
            for i in range(1, end + 1):
                A = pl[i - 1] + 1      # leftmost start index
                B = pl[i]              # max start index
                c = pl[i + z - 1]      # position of the z-th zero in this group
                nxt = pl[i + z]        # position just after that group
                count_r = nxt - c      # number of endings that keep exactly z zeros
                
                # precompute c - K and nxt - K
                cK = c - K
                nxtK = nxt - K
                
                # case 1: start l in [A .. min(B, c-K+1)] each gives the full count_r
                L1 = cK + 1
                if L1 >= A:
                    L1_end = B if B < L1 else L1
                    ans += (L1_end - A + 1) * count_r
                
                # case 2: start l in [max(A, c-K+2) .. min(B, nxt-K)]
                LS0 = cK + 2
                LS = A if A > LS0 else LS0
                RS = B if B < nxtK else nxtK
                if LS <= RS:
                    base = nxtK + 1
                    cnt = RS - LS + 1
                    # sum of arithmetic sequence D(l)=base - l over LS..RS
                    ans += ((2 * base - LS - RS) * cnt) // 2
        
        return ans