from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        ALPHA = 26

        # ---------- 1. prefix letter frequencies ----------
        # pref[ch][i]  -> how many times letter ch appears in s[0:i]
        pref = [[0]*(n+1) for _ in range(ALPHA)]
        for i, ch in enumerate(s, 1):                       # i goes from 1 .. n
            idx = ord(ch) - 97
            for c in range(ALPHA):
                pref[c][i] = pref[c][i-1]
            pref[idx][i] += 1

        # helper to obtain frequency of character k on interval [l,r] (inclusive)
        def freq(k: int, l: int, r: int) -> int:
            # l, r are valid and l<=r
            return pref[k][r+1] - pref[k][l]

        # ---------- 2. prefix mismatches for fixed pairs ----------
        mis_pref = [0]*(half+1)                 # mismatches till i-1
        mis_sum = 0
        for i in range(half):
            if s[i] != s[n-1-i]:
                mis_sum += 1
            mis_pref[i+1] = mis_sum
        total_mis = mis_pref[half]

        def miscount(l: int, r: int) -> int:    # mismatch count in [l,r] on left half
            if l > r:
                return 0
            return mis_pref[r+1] - mis_pref[l]

        # ---------- 3. process queries ----------
        res = []
        for a, b, c, d in queries:
            # mirror ranges
            jrL, jrR = n-1-b, n-1-a               # partners (on right) of A
            mbL, mbR = n-1-d, n-1-c               # partners (on left) of B

            # 3.1  fixed pairs must already match
            # indices on left half that are fixed are those outside A ∪ MirrorB
            union_mis = (miscount(a, b) +
                         miscount(mbL, mbR) -
                         miscount(max(a, mbL), min(b, mbR)) )
            if total_mis - union_mis:
                res.append(False)
                continue

            # 3.2  prepare interval intersections used often
            # intersection on right half (partners both adjustable)
            intR_L = max(jrL, c)
            intR_R = min(jrR, d)
            has_intR = intR_L <= intR_R

            # intersection on left half  (partners both adjustable)
            intL_L = max(mbL, a)
            intL_R = min(mbR, b)
            has_intL = intL_L <= intL_R

            possible = True
            rem_equal = True     # whether remaining multisets on both sides match
            for ch in range(ALPHA):
                # counts inside segments
                cntA   = freq(ch, a, b)
                cntB   = freq(ch, c, d)
                cntJR  = freq(ch, jrL, jrR)
                cntMB  = freq(ch, mbL, mbR)

                cntIntR = freq(ch, intR_L, intR_R) if has_intR else 0
                cntIntL = freq(ch, intL_L, intL_R) if has_intL else 0

                needLeft  = cntJR - cntIntR               # A must supply for pairs with fixed right
                needRight = cntMB - cntIntL               # B must supply for pairs with fixed left

                if needLeft > cntA or needRight > cntB:
                    possible = False
                    break

                remA = cntA - needLeft
                remB = cntB - needRight
                if remA != remB:
                    rem_equal = False

            res.append(possible and rem_equal)
        return res