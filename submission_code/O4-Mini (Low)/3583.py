from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        import sys
        
        # Maximum value in nums determines bounds
        maxA = max(nums)
        n = len(nums)
        
        # freq[x]: how many times x appears in nums
        freq = [0] * (maxA + 1)
        for x in nums:
            freq[x] += 1
        
        # c[d]: count of numbers divisible by d
        c = [0] * (maxA + 1)
        for d in range(1, maxA + 1):
            for m in range(d, maxA + 1, d):
                c[d] += freq[m]
        
        # tot_pairs[d]: number of pairs where both values are divisible by d
        # f[d]: number of pairs whose gcd is exactly d (via inclusion-exclusion)
        f = [0] * (maxA + 1)
        for d in range(1, maxA + 1):
            cnt = c[d]
            f[d] = cnt * (cnt - 1) // 2
        
        # Möbius inversion style sieve to get exact-gcd counts
        for d in range(maxA, 0, -1):
            mul = 2
            while d * mul <= maxA:
                f[d] -= f[d * mul]
                mul += 1
        
        # Build list of (gcd_value, count) sorted by gcd_value ascending
        gcd_list = [(g, f[g]) for g in range(1, maxA + 1) if f[g] > 0]
        
        # Sort queries by their k, keeping original index
        qs = sorted((k, i) for i, k in enumerate(queries))
        ans = [0] * len(queries)
        
        # Sweep through gcd_list and qs to answer
        total = 0
        j = 0  # pointer in gcd_list
        for k, qi in qs:
            # advance in gcd_list until we cover query index k
            while j < len(gcd_list) and total + gcd_list[j][1] <= k:
                total += gcd_list[j][1]
                j += 1
            # gcd_list[j] must be the bucket containing index k
            ans[qi] = gcd_list[j][0]
        
        return ans

# Example usage:
# sol = Solution()
# print(sol.gcdValues([2,3,4], [0,2,2]))  # [1,2,2]
# print(sol.gcdValues([4,4,2,1], [5,3,1,0]))  # [4,2,1,1]
# print(sol.gcdValues([2,2], [0,0]))  # [2,2]