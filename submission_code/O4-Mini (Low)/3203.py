from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        # partner mapping: for i in [0, half-1], partner i -> j = n-1-i
        # precompute mismatch prefix over first half
        mismatch = [0] * half
        for i in range(half):
            if s[i] != s[n - 1 - i]:
                mismatch[i] = 1
        # prefix sum of mismatches
        pm = [0] * half
        running = 0
        for i in range(half):
            running += mismatch[i]
            pm[i] = running
        
        # prefix sums of character counts over whole string
        # pc[i+1][c] = count of char c in s[:i+1]
        pc = [[0]*26 for _ in range(n+1)]
        for i, ch in enumerate(s):
            code = ord(ch) - ord('a')
            for c in range(26):
                pc[i+1][c] = pc[i][c]
            pc[i+1][code] += 1
        
        def range_mismatch(l: int, r: int) -> int:
            # sum of mismatches in [l..r] in first half
            if l > r or r < 0 or l >= half:
                return 0
            l0 = max(l, 0)
            r0 = min(r, half-1)
            if l0 > r0:
                return 0
            return pm[r0] - (pm[l0-1] if l0>0 else 0)
        
        def range_count(c: int, l: int, r: int) -> int:
            # count of char c in s[l..r]
            if l>r:
                return 0
            return pc[r+1][c] - pc[l][c]
        
        total_mismatch = pm[half-1]
        ans = []
        
        for a, b, c, d in queries:
            # 1) check fixed pairs outside L and outside R_preimage have no mismatch
            # R_preimage in first half: i such that partner[i]=j in [c,d]
            # partner[i]=n-1-i in [c,d] <=> i in [n-1-d, n-1-c]
            u = n-1-d
            v = n-1-c
            # mismatches inside L and inside R_preimage, overlap counted twice
            mL = range_mismatch(a, b)
            mRpre = range_mismatch(u, v)
            # overlap L ∩ R_preimage
            ov_l = max(a, u)
            ov_r = min(b, v)
            mOv = range_mismatch(ov_l, ov_r)
            mismatch_out = total_mismatch - mL - mRpre + mOv
            if mismatch_out != 0:
                ans.append(False)
                continue
            
            # 2) compute forced counts
            forcedL = [0]*26
            forcedR = [0]*26
            
            # for each i in L whose partner j not in R
            for i in (a,):
                pass
            # but we do via loops; we must iterate [a..b]
            for i in range(a, b+1):
                j = n-1-i
                if j < c or j > d:
                    # partner fixed, need s[j] in L
                    forcedL[ord(s[j]) - ord('a')] += 1
            # for each j in R whose partner i not in L
            for j in range(c, d+1):
                i = n-1-j
                if i < a or i > b:
                    forcedR[ord(s[i]) - ord('a')] += 1
            
            # 3) total counts in L and R
            countL = [0]*26
            countR = [0]*26
            for ch in range(26):
                countL[ch] = range_count(ch, a, b)
                countR[ch] = range_count(ch, c, d)
            
            # check forced fit
            ok = True
            for ch in range(26):
                if forcedL[ch] > countL[ch] or forcedR[ch] > countR[ch]:
                    ok = False
                    break
            if not ok:
                ans.append(False)
                continue
            
            # remaining counts
            rem = [0]*26
            for ch in range(26):
                rem[ch] = (countL[ch] - forcedL[ch]) + (countR[ch] - forcedR[ch])
                # must be even to pair among both-free
                if rem[ch] & 1:
                    ok = False
                    break
            ans.append(ok)
        
        return ans