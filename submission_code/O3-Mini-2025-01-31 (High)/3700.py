from typing import List

MOD = 10**9 + 7

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        # Compress the values so that they take indices 0..d-1
        comp_keys = sorted(set(nums))
        comp = {v: i for i, v in enumerate(comp_keys)}
        d = len(comp_keys)
        
        # Build prefix frequency array:
        # prefix[i] will be a list of length d giving the counts of each compressed value in nums[0...i-1].
        prefix = [[0] * d for _ in range(n+1)]
        for i in range(n):
            # copy previous counts
            prefix[i+1] = prefix[i][:]
            prefix[i+1][comp[nums[i]]] += 1
        
        # Build suffix frequency array:
        # suffix[i] will be counts for nums[i...n-1].
        suffix = [[0] * d for _ in range(n+1)]
        suffix[n] = [0] * d
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1][:]
            suffix[i][comp[nums[i]]] += 1
        
        # Helper function: combination C(n, r) for r=0,1,2.
        def choose(n_val: int, r: int) -> int:
            if r == 0:
                return 1
            if r == 1:
                return n_val
            if r == 2:
                return n_val*(n_val-1)//2 if n_val >=2 else 0
            return 0

        # we need the inverse of 2 modulo MOD for our mod‐division.
        inv2 = pow(2, MOD-2, MOD)
        
        ans = 0
        # The candidate (middle element) of a 5‐subsequence must have at least 2 elements before and 2 after.
        # Hence, we iterate k from 2 to n-3 (inclusive).
        for k in range(2, n - 2):
            # candidate index k; its value will be the "middle mode"
            c = comp[nums[k]]
            # Let L = counts from indices [0, k) and R = counts from indices [k+1, n)
            L = prefix[k]
            R = suffix[k+1]
            L_x = L[c]  # number of candidate X in left part
            R_x = R[c]  # number of candidate X in right part
            # non-X counts:
            L_non_total = k - L_x
            R_non_total = (n - k - 1) - R_x
            
            # We will count according to the number of additional candidate occurrences (r)
            # In a 5-element subsequence, the "middle mode" candidate appears f = r + 1 times.
            # Valid f are 2,3,4,5, i.e. r=1,2,3,4.
            # However, note that if f=2 (r=1) then the 3 non‐mode elements must all be distinct.
            #
            # We break the count into four cases: f=2, f=3, f=4 and f=5.
            
            # ----- Case f = 2 (r = 1) -----
            # Total non‐X picks = 3 (2 from the left+right combined come from non‐X)
            # and they must be all distinct (to avoid a tie of frequency 1 vs. candidate frequency 2).
            # Their positions come from the two “sides”; however the distinctness condition is “global”
            # across the two sides.
            # We will count these two splits separately:
            # (a) (rL=0, rR=1): no extra candidate X from left and one from right.
            #     Then the left must supply 2 non‐X indices (with distinct values) and
            #     the right must supply 1 non‐X index.
            #
            # (b) (rL=1, rR=0): one extra candidate X from left and none from right.
            #     Then the left supplies 1 non‐X index and the right supplies 2 non‐X indices (distinct).
            #
            # For split (a), the candidate‐part contributes: C(L_x,0) * C(R_x,1) = 1 * R_x.
            # For split (b), the candidate‐part contributes: C(L_x,1) * C(R_x,0) = L_x * 1.
            
            f2 = 0
            # Split (a): only if R_x >= 1 and left non‐X count is at least 2.
            if R_x >= 1 and L_non_total >= 2:
                # Count of ways to choose 2 indices from left non‐X with distinct values.
                # We can compute the sum over distinct keys (v != c).
                T = L_non_total
                sumL2 = 0
                for v in range(d):
                    if v == c:
                        continue
                    sumL2 += L[v] * L[v]
                # The total number of “ordered pairs” (as combination, order not mattering)
                # with distinct values is: (T^2 - Σ L[v]^2) / 2.
                pair_L = ((T * T - sumL2) % MOD) * inv2 % MOD
                
                # Now, for these left pairs the values (say a and b) are distinct.
                # For each such left pair, we must choose one non‐X index from the right
                # whose value is different from both a and b.
                # The aggregate count of valid right picks is:
                #   R_non_total * (number of left distinct pairs) 
                #   minus a correction term that subtracts, for each left pair, those
                #   right indices that have a value equal to one of the two.
                S_val = 0
                for v in range(d):
                    if v == c:
                        continue
                    # For a given value v in left non‐X, every occurrence in left paired with
                    # any occurrence from left (with v different) will disallow right indices with value v.
                    # In fact, if L[v] occurrences in left, then for each such occurrence,
                    # the number of forbidden right indices with value v is R[v].
                    # But careful: In summing over pairs, each group v will appear with a count L[v] 
                    # multiplied by (T - L[v]).
                    S_val += L[v] * R[v] * (T - L[v])
                # Then, the valid count for split (a) is:
                ways_01 = (R_non_total * pair_L - S_val) % MOD
                part_01 = ways_01 * R_x % MOD  # multiplied by candidate-part factor from right.
            else:
                part_01 = 0
            
            # Split (b): only if L_x >= 1 and R_non_total >= 2.
            if L_x >= 1 and R_non_total >= 2:
                R_total = R_non_total
                sumR2 = 0
                for v in range(d):
                    if v == c:
                        continue
                    sumR2 += R[v] * R[v]
                pair_R = ((R_total * R_total - sumR2) % MOD) * inv2 % MOD
                sum_10 = 0
                for v in range(d):
                    if v == c:
                        continue
                    # For each non‐X value v in right, the “forbidden” pairs from right (if v appears in the pair)
                    # would be R[v] * (R_total - R[v]). Subtract these from pair_R.
                    sum_10 += L[v] * (pair_R - R[v] * (R_total - R[v]))
                sum_10 %= MOD
                part_10 = (sum_10 * L_x) % MOD  # multiplied by candidate-part factor from left.
            else:
                part_10 = 0
            
            f2 = (part_01 + part_10) % MOD
            
            # ----- Case f = 3 (r = 2) ----- 
            # Now the candidate appears 3 times (1 in the middle plus 2 from the sides).
            # The non‐X parts: 2 indices total.
            # In this case there is no extra distinct‐values restriction because even if a non‐X appears twice,
            # its frequency (2) is still less than candidate frequency 3.
            # The possible splits (rL, rR) are: (0,2), (1,1) and (2,0).
            part0 = choose(R_x, 2) * choose(L_non_total, 2)
            part1 = L_x * R_x * (L_non_total) * (R_non_total)
            part2 = choose(L_x, 2) * choose(R_non_total, 2)
            f3 = (part0 + part1 + part2) % MOD
            
            # ----- Case f = 4 (r = 3) -----
            # Non‐X count is 1.
            # Splits: (1,2) and (2,1) only.
            partA = L_x * choose(R_x, 2) * (L_non_total)
            partB = choose(L_x, 2) * R_x * (R_non_total)
            f4 = (partA + partB) % MOD
            
            # ----- Case f = 5 (r = 4) -----
            # The subsequence has all elements equal to the candidate.
            f5 = choose(L_x, 2) * choose(R_x, 2)
            f5 %= MOD
            
            candidate_contrib = (f2 + f3 + f4 + f5) % MOD
            ans = (ans + candidate_contrib) % MOD
        
        return ans