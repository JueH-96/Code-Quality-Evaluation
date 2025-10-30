class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        n = len(nums)
        if n < 5:
            return 0
        
        # Coordinate-compress the values so we can store frequencies in arrays
        # rather than dictionaries of potentially large (or negative) keys.
        # We'll map each distinct value in nums to an integer in [0..d-1].
        distinct_vals = list(set(nums))
        distinct_vals.sort()
        val_to_id = {}
        for i, v in enumerate(distinct_vals):
            val_to_id[v] = i
        d = len(distinct_vals)
        comp = [val_to_id[v] for v in nums]
        
        # Precompute factorials for combinations up to n=1000
        # We will implement nCr with these precomputed factorials/inverses under MOD.
        maxN = 1000
        fact = [1]*(maxN+1)
        invFact = [1]*(maxN+1)
        for i in range(1, maxN+1):
            fact[i] = (fact[i-1]*i) % MOD
        
        # Fermat's little theorem for modular inverse of fact[i] when MOD is prime
        invFact[maxN] = pow(fact[maxN], MOD-2, MOD)
        for i in reversed(range(maxN)):
            invFact[i] = (invFact[i+1]*(i+1)) % MOD
        
        def comb(n, r):
            if r<0 or r>n:
                return 0
            return (fact[n]*invFact[r]%MOD)*invFact[n-r]%MOD
        
        # Build prefix frequency array freqVal[id][i]:
        # freqVal[id][i] = how many times value-id appears in nums[:i]  (i.e. up to but not including index i)
        freqVal = [[0]*(n+1) for _ in range(d)]
        for i in range(n):
            cid = comp[i]
            for vid in range(d):
                freqVal[vid][i+1] = freqVal[vid][i]
            freqVal[cid][i+1] += 1
        
        ans = 0
        
        # Precompute a quick helper to sum up nC2 easily if needed:
        # But we'll mostly use comb(...) directly.
        
        # Main loop: fix the middle index i3 (where 2 <= i3 <= n-3)
        # We'll count how many valid 5-subsequences have nums[i3] as unique mode.
        for i3 in range(2, n-2):
            x_id = comp[i3]
            
            # Count how many x to the left, how many x to the right
            Lx = freqVal[x_id][i3]  # occurrences of x in indices < i3
            Rx = freqVal[x_id][n] - freqVal[x_id][i3+1]  # occurrences of x in indices > i3
            # Count how many not-x to the left, not-x to the right
            L_not_x = i3 - Lx
            R_not_x = (n - 1 - i3) - Rx
            
            # ------------------
            # Case freq(x) = 5: x appears in all 5 positions
            # We need to pick 2 from the left that are x, and 2 from the right that are x.
            if Lx >= 2 and Rx >= 2:
                ways_5 = comb(Lx, 2)*comb(Rx, 2)
                ans = (ans + ways_5) % MOD
            
            # ------------------
            # Case freq(x) = 4: pick 3 from outside plus the middle => total 4 outside picks from which 3 are x and 1 is not x
            # That means (a_x + r_x) = 3, but we only have 2 picks on the left and 2 on the right.
            # So possible distributions of x: (2 on left, 1 on right) or (1 on left, 2 on right).
            # The 1 not x can come from whichever side still has a pick left.
            
            # (2 left x, 1 right x) => pick 1 from right non-x
            # Lx >= 2, Rx >= 1, R_not_x >= 1
            if Lx >= 2 and Rx >= 1 and R_not_x >= 1:
                ways_4a = comb(Lx, 2)*comb(Rx, 1)*comb(R_not_x, 1)
                ans = (ans + ways_4a) % MOD
            
            # (1 left x, 2 right x) => pick 1 from left non-x
            # Lx >= 1, Rx >= 2, L_not_x >= 1
            if Lx >= 1 and Rx >= 2 and L_not_x >= 1:
                ways_4b = comb(Lx, 1)*comb(Rx, 2)*comb(L_not_x, 1)
                ans = (ans + ways_4b) % MOD
            
            # ------------------
            # Case freq(x) = 3: pick 2 more x from left+right, plus 2 not x from left+right in total.
            # That means (a_x + r_x) = 2, (a_~x + r_~x) = 2, with a_x + a_~x=2 (left picks), r_x + r_~x=2 (right picks).
            # We enumerate (a_x, r_x) in { (0,2), (1,1), (2,0) }, then a_~x=2-a_x, r_~x=2-r_x.
            # As long as freq of not x <= 2, x=3 remains the unique mode. And with only 2 picks for "not x", we can't exceed 2 anyway.
            
            # We'll just do it by enumerating the possible distributions:
            # (0,2) => pick 0 from Lx, 2 from Rx => pick 2 from L_not_x on left, 0 from R_not_x on right
            # (1,1) => pick 1 from Lx, 1 from Rx => pick 1 from L_not_x on left, 1 from R_not_x on right
            # (2,0) => pick 2 from Lx, 0 from Rx => pick 0 from L_not_x on left, 2 from R_not_x on right
            
            ways_3 = 0
            for a_x, r_x in [(0,2), (1,1), (2,0)]:
                a_nx = 2 - a_x
                r_nx = 2 - r_x
                # check feasibility
                if Lx >= a_x and Rx >= r_x and L_not_x >= a_nx and R_not_x >= r_nx:
                    ways_3 += comb(Lx, a_x)*comb(Rx, r_x)*comb(L_not_x, a_nx)*comb(R_not_x, r_nx)
            ways_3 %= MOD
            ans = (ans + ways_3) % MOD
            
            # ------------------
            # Case freq(x) = 2: pick exactly 1 more x from the 4 outer picks,
            # plus 3 picks that are not x. We need the 3 not-x picks to be all distinct values,
            # because if any other value appears 2 or more times, it would tie freq(x)=2 and
            # x would not be the unique mode.
            # Distribution: (a_x + r_x)=1. So either (1,0) or (0,1).
            # Then the not-x distribution is (a_~x + r_~x)=3. But we also must pick exactly 2 from left side, 2 from right side.
            # So if a_x=1 => a_~x=1 => we pick 1 x + 1 not-x on the left; r_x=0 => r_~x=2 => pick 2 not-x on the right.
            # or a_x=0 => a_~x=2, r_x=1 => r_~x=1. 
            # The 3 not-x must all be distinct across left and right.
            
            # We'll build a small frequency dictionary for left side (excluding index i3) and right side,
            # excluding x. Then we do a combinational formula:
            # For (1,0):(a_~x=1,r_~x=2), the ways for x are comb(Lx,1)*comb(Rx,0) = Lx.
            # Then to pick 1 distinct value from the left side and 2 distinct values from the right side
            # (all distinct from each other), we compute:
            #   S1 = sum_{val in left-distinct} Lcount[val] * [ number of ways to pick 2 distinct from right-distinct excluding val ]
            # We'll use a "two-distinct" pairsum technique on the right to speed up.
            #
            # For (0,1):(a_~x=2,r_~x=1), ways for x = comb(Lx,0)*comb(Rx,1) = Rx.
            # Then pick 2 distinct from left-distinct plus 1 from right-distinct that is different from those 2.
            #   S2 = sum_{valR in right-distinct} Rcount[valR] * [ number of ways to pick 2 distinct from left-distinct excluding valR ]
            
            # Let's build the needed info for left-distinct and right-distinct quickly.
            
            # If there are not enough not-x on the left or right for these distributions, skip.
            if L_not_x >= 1 and R_not_x >= 2 and Lx > 0:
                # Build frequency dictionaries for left and right (excluding x)
                # left side is nums[0..i3-1], right side is nums[i3+1..n-1]
                # But we already have freqVal. We'll iterate over distinct ids to collect counts.
                # We'll do it once for left & right. (We might do it only once per i3 because
                # we also need it for the other distribution.)
                
                # We can cache these in lists or dictionaries:
                # Lcount_id[val_id] = freq of val_id on the left side
                # Rcount_id[val_id] = freq of val_id on the right side
                # Then exclude x_id in usage.
                
                # We'll only build them once per i3:
                # But to handle both subcases of freq(x)=2, we'll do it once and reuse.
                
                # To avoid rebuilding multiple times, let's do it outside the condition checks:
                pass
            
            # We'll build them once (l_dict, r_dict) for not-x. Then we can do sum-of-squares approach.
            # But we do it only once if we haven't done it yet. Let's just do it once here:
            
            # Build left_counts[vid] for vid != x_id
            left_counts = {}
            for vid in range(d):
                cnt = freqVal[vid][i3]
                if vid == x_id or cnt == 0:
                    continue
                left_counts[vid] = cnt
            # Build right_counts[vid] for vid != x_id
            right_counts = {}
            for vid in range(d):
                cnt = freqVal[vid][n] - freqVal[vid][i3+1]
                if vid == x_id or cnt == 0:
                    continue
                right_counts[vid] = cnt
            
            # Precompute sumOfLeft = total not-x on left, sumSqLeft for the squares
            sumOfLeft = 0
            sumSqLeft = 0
            for v in left_counts.values():
                sumOfLeft += v
                sumSqLeft += v*v
            # Precompute sumOfRight, sumSqRight
            sumOfRight = 0
            sumSqRight = 0
            for v in right_counts.values():
                sumOfRight += v
                sumSqRight += v*v
            
            sumOfLeft_mod = sumOfLeft % MOD
            sumSqLeft_mod = sumSqLeft % MOD
            sumOfRight_mod = sumOfRight % MOD
            sumSqRight_mod = sumSqRight % MOD
            
            # totalPairsLeft = sum_{p<q} Lcount[p]*Lcount[q] 
            # = ( (Σ Lcount[p])^2 - Σ(Lcount[p]^2 ) ) / 2
            totalPairsLeft = ( (sumOfLeft_mod*sumOfLeft_mod - sumSqLeft_mod) % MOD ) * pow(2, MOD-2, MOD)
            totalPairsLeft %= MOD
            # totalPairsRight similarly
            totalPairsRight = ( (sumOfRight_mod*sumOfRight_mod - sumSqRight_mod) % MOD ) * pow(2, MOD-2, MOD)
            totalPairsRight %= MOD
            
            # pairSumLeft[val] = Lcount[val]*(sumOfLeft - Lcount[val]) in normal integer
            # We'll store in a dict. (Do it in mod later, but we also need normal for small differences.)
            pairSumLeft = {}
            for v, c in left_counts.items():
                pairSumLeft[v] = c*(sumOfLeft - c)
            # pairSumRight[val] similarly
            pairSumRight = {}
            for v, c in right_counts.items():
                pairSumRight[v] = c*(sumOfRight - c)
            
            # Now compute freq(x)=2 distribution sums:
            
            # subcase A: (a_x=1, r_x=0) => picks = comb(Lx,1)*comb(Rx,0) = Lx
            # pick 1 distinct value from left, 2 distinct from right (none of which match the left).
            # Ways for the distinct picks in the not-x part:
            # S1 = ∑_{val in left_counts} [ Lcount[val] * ( #ways to pick 2 distinct from right_counts excluding val ) ]
            # (#ways to pick 2 distinct from right_counts excluding val ) =
            #   totalPairsRight - pairSumRight[val] if val in right_counts,
            #   else totalPairsRight if val not in right_counts.
            
            S1 = 0
            for val, lc in left_counts.items():
                # ways to pick from the right
                ps = pairSumRight[val] if val in pairSumRight else 0
                # normal integer arithmetic
                # number of pairs in right excluding val = totalPairsRight_int - ps
                # but we have totalPairsRight in mod form. We also have ps in normal int form.
                # We'll do this carefully in mod, but we need the difference in normal int form first,
                # then mod it.
                
                # We'll define totalPairsRight_int outside to handle big integer:
                # but totalPairsRight is modded. We also stored sumOfRight and sumSqRight in normal integer
                # pairSumRight[val] is in normal int. We can re-compute or do carefully.
                
                # Easiest is to compute the "excluded" pairs in normal integer, then mod at the end:
                # totalPairsRight_int = sum_{p<q} right_counts[p]*right_counts[q]
                # we can just do totalPairsRight_int = sumOfRight*(sumOfRight-1)//2 - ...
                # but that might be large. We'll do it systematically:
                # We'll do a direct formula for "pairs that exclude val": 
                # pairs(R) - Rcount[val]*(sumOfRight - Rcount[val]) / 2 is the count. But let's do it carefully:
                
                # Actually let's do a small function that returns the total pairs in normal int:
                # Because sumOfRight can be up to ~1000, the product can be up to 1e6, which is safe in Python int.
                
                # We'll just build it once (no big overhead). For correctness and clarity:
                # NOTE: we've already built sumOfRight, sumSqRight. So totalPairsRight_int = (sumOfRight^2 - sumSqRight)//2
                totalPairsRight_int = (sumOfRight*sumOfRight - sumSqRight)//2
                ps_val = pairSumRight[val] if val in pairSumRight else 0
                # The number of pairs that involve val is ps_val.
                # So excluding val => totalPairsRight_int - ps_val.
                # Then multiply by lc for the left pick.
                
                rightDistinctPairsExVal = totalPairsRight_int - ps_val
                # add to S1
                S1 += lc * rightDistinctPairsExVal
            
            # subcase B: (a_x=0, r_x=1) => picks = comb(Lx,0)*comb(Rx,1) = Rx
            # pick 2 distinct from left, 1 distinct from right that is different from the 2 on the left.
            # S2 = ∑_{valR in right_counts} [ Rcount[valR] * (#ways to pick 2 distinct from left_counts excluding valR) ]
            
            S2 = 0
            # totalPairsLeft_int
            totalPairsLeft_int = (sumOfLeft*sumOfLeft - sumSqLeft)//2
            for valR, rc in right_counts.items():
                ps_valR = pairSumLeft[valR] if valR in pairSumLeft else 0
                leftDistinctPairsExValR = totalPairsLeft_int - ps_valR
                S2 += rc * leftDistinctPairsExValR
            
            # Now multiply by the ways to pick x:
            ways_2 = 0
            # (1,0) => Lx * S1
            ways_2 += Lx * S1
            # (0,1) => Rx * S2
            ways_2 += Rx * S2
            ways_2 %= MOD
            
            ans = (ans + ways_2) % MOD
        
        return ans % MOD