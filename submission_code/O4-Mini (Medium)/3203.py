from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        # Build prefix mismatch counts for pairs (i, n-1-i) in left half [0..half-1]
        mismatch_ps = [0] * half
        for i in range(half):
            mis = 1 if s[i] != s[n-1-i] else 0
            mismatch_ps[i] = mis + (mismatch_ps[i-1] if i > 0 else 0)
        total_mismatch = mismatch_ps[half-1]
        
        # Build prefix character counts for s
        # prefix_count[c][i] = count of char c in s[0..i-1]
        prefix_count = [[0] * (n+1) for _ in range(26)]
        for i, ch in enumerate(s):
            ci = ord(ch) - 97
            for c in range(26):
                prefix_count[c][i+1] = prefix_count[c][i]
            prefix_count[ci][i+1] += 1
        
        # helper: count of char c in s[l..r], inclusive
        def cnt_char(c: int, l: int, r: int) -> int:
            # if invalid range
            if l > r:
                return 0
            return prefix_count[c][r+1] - prefix_count[c][l]
        
        # helper: count array of all 26 chars in s[l..r]
        def cnt_range(l: int, r: int) -> List[int]:
            res = [0] * 26
            if l > r:
                return res
            # r+1 and l safe
            for c in range(26):
                res[c] = prefix_count[c][r+1] - prefix_count[c][l]
            return res
        
        # helper: mismatch count in i-range [l..r] in left half
        def cnt_mis(l: int, r: int) -> int:
            if l > r:
                return 0
            return mismatch_ps[r] - (mismatch_ps[l-1] if l > 0 else 0)
        
        ans = []
        for (a, b, c, d) in queries:
            # L window in left half = [a,b], R window in right half = [c,d]
            # Corresponding i-range whose j in R is L2 = [n-1-d .. n-1-c]
            L2s = n - 1 - d
            L2e = n - 1 - c
            
            # 1) Check fixed mismatches (A positions)
            # U = union of L and L2 in left-half index space
            # mismatches in U:
            mis_L   = cnt_mis(a, b)
            mis_L2  = cnt_mis(L2s, L2e)
            # intersection of L and L2
            i1 = max(a, L2s)
            i2 = min(b, L2e)
            mis_inter = cnt_mis(i1, i2)
            mis_U = mis_L + mis_L2 - mis_inter
            # mismatches in A = total_mismatch - mismatches in U
            if total_mismatch - mis_U > 0:
                ans.append(False)
                continue
            
            # 2) Gather inventories
            # initial L and R inventory
            L_init = cnt_range(a, b)
            R_init = cnt_range(c, d)
            
            # 3) Count B demands (i fixed, j in R) => sum over i in (L2 \ L), want s[i]
            B = [0] * 26
            # interval 1: [L2s .. min(L2e, a-1)]
            if L2s <= a-1:
                r1 = min(L2e, a-1)
                tmp = cnt_range(L2s, r1)
                for x in range(26):
                    B[x] += tmp[x]
            # interval 2: [max(b+1, L2s) .. L2e]
            if b+1 <= L2e:
                l2 = max(b+1, L2s)
                tmp = cnt_range(l2, L2e)
                for x in range(26):
                    B[x] += tmp[x]
            
            # 4) Count C demands (i in L, j fixed) => sum over i in (L \ L2), want s[j]
            C = [0] * 26
            # interval 1: [a .. min(b, L2s-1)]
            if a <= L2s-1:
                r1 = min(b, L2s-1)
                # j in [n-1-r1 .. n-1-a]
                j_lo = n-1-r1
                j_hi = n-1-a
                tmp = cnt_range(j_lo, j_hi)
                for x in range(26):
                    C[x] += tmp[x]
            # interval 2: [max(a, L2e+1) .. b]
            if L2e+1 <= b:
                l2 = max(a, L2e+1)
                # j in [n-1-b .. n-1-l2]
                j_lo = n-1-b
                j_hi = n-1-l2
                tmp = cnt_range(j_lo, j_hi)
                for x in range(26):
                    C[x] += tmp[x]
            
            # 5) Check inventory feasibility for B and C
            ok = True
            for x in range(26):
                if B[x] > R_init[x] or C[x] > L_init[x]:
                    ok = False
                    break
            if not ok:
                ans.append(False)
                continue
            
            # 6) Compute remainders and compare
            # rem_L = L_init - C, rem_R = R_init - B
            for x in range(26):
                L_init[x] -= C[x]
                R_init[x] -= B[x]
                if L_init[x] != R_init[x]:
                    ok = False
                    break
            ans.append(ok)
        
        return ans