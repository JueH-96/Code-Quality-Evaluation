from typing import List
from bisect import bisect_left

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        max_num = max(nums)
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1

        # Compute c[g]: number of elements divisible by g
        c = [0] * (max_num + 1)
        for g in range(1, max_num + 1):
            for k in range(g, max_num + 1, g):
                c[g] += freq[k]

        # Compute count[g] = number of pairs with GCD exactly g
        count = [0] * (max_num + 1)
        for g in range(1, max_num + 1):
            count[g] = c[g] * (c[g] - 1) // 2

        # Apply inclusion-exclusion to subtract multiples
        for g in range(max_num, 0, -1):
            k = 2 * g
            while k <= max_num:
                count[g] -= count[k]
                k += g

        # Build sorted list of gcd values and their prefix sums
        sorted_gcds = []
        prefix = []
        current_sum = 0
        for g in range(1, max_num + 1):
            if count[g] > 0:
                sorted_gcds.append(g)
                current_sum += count[g]
                prefix.append(current_sum)

        # Answer each query using binary search on prefix array
        res = []
        for q in queries:
            left, right = 0, len(prefix) - 1
            ans = -1
            while left <= right:
                mid = (left + right) // 2
                if prefix[mid] >= q + 1:
                    ans = mid
                    right = mid - 1
                else:
                    left = mid + 1
            res.append(sorted_gcds[ans])
        return res