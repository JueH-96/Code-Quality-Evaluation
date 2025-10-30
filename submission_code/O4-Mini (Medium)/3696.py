from math import gcd

class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        # cnt[M] will be a list of length M, where cnt[M][r] is the number of substrings
        # ending at the previous position whose value mod M == r.
        cnt = {M: [0] * M for M in range(1, 10)}
        
        ans = 0
        for ch in s:
            digit = int(ch)
            # First, if digit != 0, add counts of substrings ending at previous pos
            # whose X mod M == 0, plus one for the empty-X case (single-digit substring).
            if digit != 0:
                d = digit
                m = 10 % d
                g = gcd(m, d)
                M = d // g
                # cnt[M][0] is the number of non-empty X substrings divisible by M
                ans += cnt[M][0] + 1
            
            # Now update cnt for each modulus M = 1..9 to account for substrings
            # extended by this digit, and the new one-digit substring.
            for M in range(1, 10):
                old = cnt[M]
                new = [0] * M
                # extend all previous substrings ending at last position
                # by appending 'digit'
                for r in range(M):
                    c = old[r]
                    if c:
                        r2 = (r * 10 + digit) % M
                        new[r2] += c
                # start a new substring at this position
                r_start = digit % M
                new[r_start] += 1
                cnt[M] = new
        
        return ans