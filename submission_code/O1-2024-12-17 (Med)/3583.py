from typing import List
from bisect import bisect_left

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # ---------------------------------------------------------------------
        # We need to form all pairs (i < j) of nums, compute their GCDs,
        # sort these GCDs, and answer queries about the element at a given index
        # in the sorted-gcd array. However, directly enumerating all pairs is
        # impossible for large n (as n*(n-1)/2 can be very large).
        #
        # Instead, we use a known technique:
        #   1) The GCD of any pair cannot exceed max(nums). Let M = max(nums).
        #   2) Count how many elements of nums are exactly k in a freq array (k = 1..M).
        #   3) For each d = 1..M, let cnt[d] = number of elements in nums divisible by d.
        #   4) The number of pairs whose GCD is a multiple of d is comb(cnt[d], 2).
        #   5) To find how many pairs have GCD exactly d, we use:
        #        c[d] = pairCnt[d] - sum_{k=2..∞, k*d <= M} c[k*d]
        #      where pairCnt[d] = comb(cnt[d], 2).
        #   6) We then know c[d] is exactly the count of pairs having gcd = d.
        #   7) Build a prefix sum S so that S[d] = c[1] + c[2] + ... + c[d].
        #   8) For each query q (0-based index), we find the unique d such that
        #      S[d-1] <= q < S[d]. That means gcdPairs[q] = d.
        #      We do this via a binary search on the prefix sum array.
        #
        # This approach avoids explicitly sorting all pairs. 
        # ---------------------------------------------------------------------
        
        # Edge case: if nums has length < 2 (but per constraints, n >= 2)
        if len(nums) < 2:
            return []
        
        # 1) Find the maximum value of nums (since gcd can't exceed max(nums))
        M = max(nums)
        
        # 2) Build freq array where freq[x] = number of times x appears in nums
        freq = [0] * (M + 1)
        for x in nums:
            freq[x] += 1
        
        # 3) Build cnt array where cnt[d] = number of elements divisible by d
        cnt = [0] * (M + 1)
        for d in range(1, M + 1):
            s = 0
            for multiple in range(d, M + 1, d):
                s += freq[multiple]
            cnt[d] = s
        
        # Helper function for nC2
        def comb2(x):
            return x * (x - 1) // 2 if x >= 2 else 0
        
        # pairCnt[d] = number of pairs (i<j) where both elements are multiples of d
        pairCnt = [0] * (M + 1)
        for d in range(1, M + 1):
            pairCnt[d] = comb2(cnt[d])
        
        # 4) Compute c[d] = number of pairs with gcd exactly d
        #    using the principle of inclusion-exclusion (descending order)
        c = [0] * (M + 1)
        for d in range(M, 0, -1):
            # Start with all pairs whose gcd is a multiple of d
            val = pairCnt[d]
            # Subtract pairs counted in multiples of d (2d, 3d, ...)
            multiple = 2 * d
            while multiple <= M:
                val -= c[multiple]
                multiple += d
            c[d] = val
        
        # 5) Build prefix sums S, where S[d] = sum(c[1]..c[d])
        S = [0] * (M + 1)
        for i in range(1, M + 1):
            S[i] = S[i - 1] + c[i]
        
        # 6) For each query, we want gcdPairs[q] (0-based).
        #    Find smallest d such that S[d] > q, i.e. S[d] > q => gcdPairs[q] = d.
        #    We do a binary search on S.
        ans = []
        for q in queries:
            # We look for index d with S[d] > q
            # bisect_left(S, q+1) finds the leftmost d s.t. S[d] >= q+1,
            # which implies S[d] > q. That d is our gcd value.
            d = bisect_left(S, q + 1)
            ans.append(d)
        
        return ans