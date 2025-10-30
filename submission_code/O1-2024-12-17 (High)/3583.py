class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        from bisect import bisect_left
        
        # 1) Determine the maximum value in nums for array sizing
        maxA = max(nums)
        
        # 2) Build a frequency array for values up to maxA
        freq = [0] * (maxA + 1)
        for x in nums:
            freq[x] += 1
        
        # 3) Compute how many elements are multiples of each d (1 <= d <= maxA)
        multiple = [0] * (maxA + 1)
        for d in range(1, maxA + 1):
            s = 0
            for m in range(d, maxA + 1, d):
                s += freq[m]
            multiple[d] = s
        
        # 4) pair_count[d] = # of pairs whose GCD is a multiple of d
        #    = C(multiple[d], 2)
        pair_count = [0] * (maxA + 1)
        for d in range(1, maxA + 1):
            c = multiple[d]
            pair_count[d] = c * (c - 1) // 2
        
        # 5) gcd_count[d] = number of pairs whose GCD is exactly d
        #    We use a classic inclusion-exclusion approach, going in descending order:
        #    gcd_count[d] = pair_count[d] - sum of gcd_count[k*d] for k >= 2
        gcd_count = [0] * (maxA + 1)
        for d in range(maxA, 0, -1):
            tmp = pair_count[d]
            for k in range(2 * d, maxA + 1, d):
                tmp -= gcd_count[k]
            gcd_count[d] = tmp
        
        # 6) Build a prefix sum array (in ascending gcd order) for quick index lookups
        unique_ds = []
        gcd_cum = [0]
        for d in range(1, maxA + 1):
            c = gcd_count[d]
            if c > 0:
                unique_ds.append(d)
                gcd_cum.append(gcd_cum[-1] + c)
        
        # 7) For each query q, find which gcd value corresponds to the 0-based index q.
        #    We do a binary search in gcd_cum to locate the appropriate gcd bucket.
        ans = []
        for q in queries:
            # Find smallest i where gcd_cum[i] > q
            i = bisect_left(gcd_cum, q + 1)
            ans.append(unique_ds[i - 1])
        
        return ans