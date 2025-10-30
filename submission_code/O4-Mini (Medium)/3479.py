class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        # prefixOnes[i] = number of '1's in s[:i]
        prefixOnes = [0] * (n + 1)
        for i, ch in enumerate(s):
            prefixOnes[i+1] = prefixOnes[i] + (ch == '1')
        # positions of zeros, with sentinels at -1 and n
        P = [-1]
        for i, ch in enumerate(s):
            if ch == '0':
                P.append(i)
        P.append(n)
        zeros_count = len(P) - 2
        
        ans = 0
        # Case z = 0 (no zeros): count all-1 substrings
        # runs of consecutive '1's
        run = 0
        for ch in s:
            if ch == '1':
                run += 1
            else:
                if run > 0:
                    ans += run * (run + 1) // 2
                run = 0
        if run > 0:
            ans += run * (run + 1) // 2
        
        # Maximum z to consider
        import math
        K = int(math.isqrt(n))
        # Cases z >= 1
        for z in range(1, K+1):
            if z > zeros_count:
                break
            z2 = z*z
            # slide a window of size z over the zero positions P[1..zeros_count]
            # window indices in P are i .. i+z-1, with i from 1 to zeros_count-z+1
            for i in range(1, zeros_count - z + 2):
                # gaps on left and right of this zero-window
                leftGap = P[i] - P[i-1] - 1
                rightGap = P[i+z] - P[i+z-1] - 1
                # ones inside the window of zeros (between first and last zero inclusive)
                # = prefixOnes[P[i+z-1]+1] - prefixOnes[P[i]]
                midOnes = prefixOnes[P[i+z-1] + 1] - prefixOnes[P[i]]
                need = z2 - midOnes
                total = (leftGap + 1) * (rightGap + 1)
                if need <= 0:
                    # already enough ones inside, all extensions valid
                    ans += total
                else:
                    # need x+y >= need, where 0<=x<=leftGap, 0<=y<=rightGap
                    L, R = leftGap, rightGap
                    if need > L + R:
                        # even max extension not enough
                        continue
                    # count bad pairs with x+y < need
                    # bad = sum_{x=0..u} (min(R, need-1-x) + 1)
                    # where u = min(L, need-1)
                    u = min(L, need - 1)
                    a0 = (need - 1) - R
                    if a0 < 0:
                        c = -1
                    else:
                        c = min(u, a0)
                    # part1: x=0..c => each contributes (R+1)
                    if c >= 0:
                        part1 = (c + 1) * (R + 1)
                    else:
                        part1 = 0
                    # part2: x=c+1..u => contributes (need - x) each
                    if u >= c + 1:
                        n2 = u - c
                        # sum_{x=c+1..u} (need - x) = n2*need - sum_{x=c+1..u} x
                        # sum x = ( (c+1) + u ) * n2 / 2
                        sum_x = (c + 1 + u) * n2 // 2
                        part2 = n2 * need - sum_x
                    else:
                        part2 = 0
                    bad = part1 + part2
                    ans += total - bad
        return ans