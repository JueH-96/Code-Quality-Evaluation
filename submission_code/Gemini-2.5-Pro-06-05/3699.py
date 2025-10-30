import collections
from math import gcd
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        """
        Calculates the number of special subsequences in an array of positive integers.

        A special subsequence is of length 4, with indices (p, q, r, s) such that:
        1. p < q < r < s
        2. nums[p] * nums[r] == nums[q] * nums[s]
        3. q - p > 1, r - q > 1, s - r > 1

        The method uses dynamic programming to solve the problem efficiently.
        The core idea is to rephrase the condition as a ratio equality:
        nums[p] / nums[q] == nums[s] / nums[r].

        The algorithm proceeds in three main steps:
        1. Pre-computation of (r, s) pairs:
           For each possible index r, we compute a map `dp_rs[r]` that stores the counts 
           of all reduced ratios (nums[s]/g, nums[r]/g) for all valid s (s > r + 1).

        2. Suffix sum computation:
           We then compute a suffix sum map `dp_rs_suffix_sum`. `dp_rs_suffix_sum[i]`
           stores the total counts of ratios for all pairs (r, s) where r >= i.
           This allows for O(1) lookup of the total number of valid (r, s) pairs
           starting from a certain index.

        3. Main counting loop:
           We iterate through all valid (p, q) pairs. For each pair, we calculate its
           ratio. The number of matching (r, s) pairs that can complete the subsequence
           is then retrieved from our suffix sum map. Summing these up gives the total count.
        """
        n = len(nums)
        if n < 7:
            return 0

        # dp_rs[r] maps ratio -> count for pairs (r, s) where ratio is 
        # (nums[s]/g, nums[r]/g) and s > r + 1.
        dp_rs = [collections.defaultdict(int) for _ in range(n)]
        for r in range(n):
            val_r = nums[r]
            # s must be at least r + 2 to satisfy s - r > 1.
            for s in range(r + 2, n):
                val_s = nums[s]
                common_divisor = gcd(val_s, val_r)
                ratio = (val_s // common_divisor, val_r // common_divisor)
                dp_rs[r][ratio] += 1

        # dp_rs_suffix_sum[i] stores the sum of dp_rs[k] for all k >= i.
        # It maps a ratio to the total count of (r, s) pairs with that ratio, where r >= i.
        dp_rs_suffix_sum = [collections.defaultdict(int) for _ in range(n + 1)]
        for r in range(n - 1, -1, -1):
            map_r = dp_rs[r]
            map_r_plus_1 = dp_rs_suffix_sum[r + 1]
            
            # To avoid large copy overhead, we intelligently merge the two maps.
            # We copy the larger map and iterate through the smaller one.
            if len(map_r) < len(map_r_plus_1):
                dp_rs_suffix_sum[r] = map_r_plus_1.copy()
                for ratio, count in map_r.items():
                    dp_rs_suffix_sum[r][ratio] += count
            else:
                dp_rs_suffix_sum[r] = map_r.copy()
                for ratio, count in map_r_plus_1.items():
                    dp_rs_suffix_sum[r][ratio] += count

        total_count = 0
        # Iterate through all possible (p, q) pairs satisfying p < q - 1.
        for q in range(2, n):
            val_q = nums[q]
            # p can range from 0 to q-2.
            for p in range(q - 1):
                val_p = nums[p]
                
                # The equation is nums[p] * nums[r] == nums[q] * nums[s],
                # which rearranges to nums[s] / nums[r] == nums[p] / nums[q].
                # We calculate the reduced ratio for (p, q) to find matching (s, r) pairs.
                common_divisor = gcd(val_p, val_q)
                search_ratio = (val_p // common_divisor, val_q // common_divisor)
                
                # We need to find pairs (r, s) satisfying:
                # 1. r - q > 1  =>  r >= q + 2
                # 2. s - r > 1
                # 3. The ratio condition: nums[s]/nums[r] == nums[p]/nums[q].
                
                # The count of such pairs (r, s) with r >= q + 2 is precisely
                # stored in our precomputed suffix sum map.
                if q + 2 < n:
                    count_rs = dp_rs_suffix_sum[q + 2].get(search_ratio, 0)
                    total_count += count_rs
                    
        return total_count