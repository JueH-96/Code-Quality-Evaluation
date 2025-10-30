from typing import List
from bisect import bisect_left

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        max_num = max(nums)
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1
        
        # Compute cnt[g]: number of elements divisible by g
        cnt = [0] * (max_num + 1)
        for g in range(1, max_num + 1):
            for multiple in range(g, max_num + 1, g):
                cnt[g] += freq[multiple]
        
        # Compute C(g): number of pairs where both are divisible by g
        C = [0] * (max_num + 1)
        for g in range(1, max_num + 1):
            C[g] = cnt[g] * (cnt[g] - 1) // 2
        
        # Compute F(g): number of pairs where GCD is exactly g
        F = [0] * (max_num + 1)
        for g in range(max_num, 0, -1):
            F[g] = C[g]
            multiple = 2 * g
            while multiple <= max_num:
                F[g] -= F[multiple]
                multiple += g
        
        # Build sorted list of (g, F(g)) where F(g) > 0
        sorted_gcd = []
        for g in range(1, max_num + 1):
            if F[g] > 0:
                sorted_gcd.append((g, F[g]))
        
        # Build prefix sum of F(g)
        prefix_sum = []
        current_sum = 0
        for g, f in sorted_gcd:
            current_sum += f
            prefix_sum.append(current_sum)
        
        # Answer queries
        answer = []
        for q in queries:
            # Find the smallest index where prefix_sum[index] >= q + 1
            idx = bisect_left(prefix_sum, q + 1)
            if idx < len(prefix_sum):
                answer.append(sorted_gcd[idx][0])
            else:
                # In case query index is out of bounds
                answer.append(-1)
        
        return answer