from typing import List
from collections import Counter, defaultdict

MOD = 10 ** 9 + 7


class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)

        # frequency of every value in the suffix (indices 0 .. n-1  initially)
        right = Counter(nums)
        # frequency of every value in the prefix (empty at the start)
        left = defaultdict(int)

        # duplicates counters  ( Σ C(cnt,2)  for each side )
        dup_right = sum(c * (c - 1) // 2 for c in right.values())
        dup_left = 0

        # cross side helper sums
        # S1  = Σ L[v] * R[v]
        # S2  = Σ L[v] * (R[v])²
        # S3  = Σ R[v] * (L[v])²
        S1 = S2 = S3 = 0

        ans = 0

        def add_contribution(lv: int, rv: int):
            nonlocal S1, S2, S3
            S1 += lv * rv
            S2 += lv * rv * rv
            S3 += rv * lv * lv

        def remove_contribution(lv: int, rv: int):
            nonlocal S1, S2, S3
            S1 -= lv * rv
            S2 -= lv * rv * rv
            S3 -= rv * lv * lv

        for j in range(n):
            m = nums[j]

            lv = left[m]          # occurrences of m on the left
            rv = right[m]         # occurrences of m on the right (includes index j now)

            # -------------   step 1 : move index j from right side ----------------
            remove_contribution(lv, rv)

            # update duplicate counter of the right part (remove one occurrence)
            if rv >= 2:
                dup_right -= (rv - 1)          # C(rv,2) - C(rv-1,2) == rv-1
            rv -= 1
            if rv == 0:
                del right[m]
            else:
                right[m] = rv

            add_contribution(lv, rv)            # new (lv, rv) after removal from right

            # ------------------   compute contribution for this pivot -------------
            LM, RM = lv, rv
            left_size = j
            right_size = n - j - 1
            Ln = left_size - LM                 # non–m elements on the left
            Rn = right_size - RM                # non–m elements on the right

            # duplicates among non-m values
            dup_left_nonm = dup_left - LM * (LM - 1) // 2
            dup_right_nonm = dup_right - RM * (RM - 1) // 2

            # number of pairs of DISTINCT values (two indices) not equal to m
            TL = max(0, Ln * (Ln - 1) // 2 - dup_left_nonm)
            TR = max(0, Rn * (Rn - 1) // 2 - dup_right_nonm)

            # ways to pick the 2 indices on every side according to
            # a = #m on the left,  b = #m on the right
            lw0 = Ln * (Ln - 1) // 2 if Ln >= 2 else 0           # a = 0
            lw1 = LM * Ln if LM >= 1 and Ln >= 1 else 0          # a = 1
            lw2 = LM * (LM - 1) // 2 if LM >= 2 else 0           # a = 2

            rw0 = Rn * (Rn - 1) // 2 if Rn >= 2 else 0           # b = 0
            rw1 = RM * Rn if RM >= 1 and Rn >= 1 else 0          # b = 1
            rw2 = RM * (RM - 1) // 2 if RM >= 2 else 0           # b = 2

            lw0 %= MOD
            lw1 %= MOD
            lw2 %= MOD
            rw0 %= MOD
            rw1 %= MOD
            rw2 %= MOD

            # -------------------   cases with k >= 3 (a+b >= 2) -------------------
            contrib_kge3 = (
                lw2 * ((rw0 + rw1 + rw2) % MOD) +
                lw1 * ((rw1 + rw2) % MOD) +
                lw0 * rw2
            ) % MOD
            ans = (ans + contrib_kge3) % MOD

            # ---------------   preparations for k == 2 computations ---------------
            # cross sums WITHOUT the pivot value m
            S1_ex = S1 - LM * RM
            S2_ex = S2 - LM * RM * RM
            S3_ex = S3 - RM * LM * LM

            # ----------------------   (a = 1 , b = 0) -----------------------------
            # Needs: LM >=1 , Ln >=1 , Rn >=2
            if LM and Ln and Rn >= 2 and TR:
                seq1 = Ln * TR - (Rn * S1_ex - S2_ex)
                seq1 %= MOD
                ans = (ans + LM * seq1) % MOD

            # ----------------------   (a = 0 , b = 1) -----------------------------
            # Needs: RM >=1 , Rn >=1 , Ln >=2
            if RM and Rn and Ln >= 2 and TL:
                seq2 = TL * Rn - (Ln * S1_ex - S3_ex)
                seq2 %= MOD
                ans = (ans + RM * seq2) % MOD

            # -----------------   step 3 : move index j to left side ----------------
            remove_contribution(lv, rv)          # old contribution (lv , rv)

            dup_left += lv                       # C(lv+1,2) - C(lv,2) == lv
            lv += 1
            left[m] = lv

            add_contribution(lv, rv)             # new contribution (lv , rv)

        return ans % MOD