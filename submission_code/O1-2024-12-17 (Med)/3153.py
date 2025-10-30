class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        """
        We have an array nums and an integer k. We can perform any number of
        the following operations:

            Pick distinct indices i and j, and simultaneously set:
                nums[i] = (old_nums[i] & old_nums[j])
                nums[j] = (old_nums[i] | old_nums[j])

        Eventually, we will choose k elements from the final array and maximize
        the sum of their squares. We return that maximum sum of squares modulo 1e9+7.

        ----------------------------------------------------------------------
        INSIGHT / SOLUTION EXPLANATION:

        Let p_b be the count of how many elements in nums have bit b set
        (where b ranges over all possible bit positions, up to ~30 for 10^9).

        Under the allowed operations, a bit originally present in p_b elements
        cannot end up in more than p_b elements at the end (we cannot replicate
        a bit beyond its initial number of occurrences).  However, we can move
        or combine bits to create larger-valued elements.

        Ultimately, we only care about picking k elements (not necessarily
        distinct from the perspective of bits) that maximize the sum of squares.

        A key fact is that for each bit b, we may choose to place it in some
        r_b (0 <= r_b <= min(k, p_b)) of the chosen k elements.  Similarly,
        for each pair of bits (b, c) within the same chosen element, we gain
        a positive "cross-term" = 2^(b+c+1) in that element's square.

        It turns out the optimal arrangement is:
          - For each bit b, use it in r_b = min(k, p_b) of the chosen elements.
          - For every pair of bits (b, c) that we use, we place them together
            in exactly min(r_b, r_c) of those elements to maximize cross-terms.

        Then the sum of squares is:
            SUM_over_b [ r_b * 2^(2b) ] 
          + SUM_over_(b < c) [ min(r_b, r_c) * 2^(b+c+1) ].

        Where r_b = min(k, p_b).

        We compute p_b for b=0..30 (or 31), set r_b, and then sum up the terms
        carefully under modulo 1e9+7.  This yields the required maximum sum
        of squares.
        """

        import sys
        MOD = 10**9 + 7

        # 1) Count p_b = how many numbers have the b-th bit set
        max_bit = 31  # 2^30 = 1073741824, enough for up to 10^9
        pb = [0]*(max_bit+1)
        for x in nums:
            for b in range(max_bit+1):
                if x & (1 << b):
                    pb[b] += 1
        
        # 2) For each bit b, we will use it in r_b = min(k, p_b[b]) of the chosen k elements
        rb = [min(k, pb[b]) for b in range(max_bit+1)]
        
        # Precompute powers of 2 up to 2^(2*max_bit+1) under modulo
        # (we need up to 2^(60+1) = 2^61 possibly, so let's go a bit higher)
        limit_pow = 2*(max_bit+1)
        pow2 = [1]*(limit_pow+2)
        for i in range(1, limit_pow+2):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        # Helper to get 2^x mod M safely, x up to around 61 or 62
        def two_power(x):
            return pow2[x] if x <= limit_pow+1 else pow(pow2[limit_pow+1], 1 << (x-(limit_pow+1)), MOD)
        
        ans = 0
        
        # 3a) Add the single-bit-square terms: sum_{b} (r_b * 2^(2*b))
        for b in range(max_bit+1):
            if rb[b] > 0:
                # 2^(2b) mod M
                val_sq = two_power(2*b)
                ans = (ans + rb[b]*val_sq) % MOD
        
        # 3b) Add the pairwise synergy: sum_{b<c} min(r_b, r_c)*2^(b+c+1)
        for b in range(max_bit+1):
            for c in range(b+1, max_bit+1):
                if rb[b] > 0 and rb[c] > 0:
                    s = min(rb[b], rb[c])
                    # Contribution: s * 2^(b+c+1)
                    pair_val = two_power(b + c + 1)
                    ans = (ans + s*pair_val) % MOD
        
        return ans % MOD