from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        from math import gcd
        from collections import defaultdict
        from bisect import bisect_right

        max_num = max(nums)
        freq = [0] * (max_num + 2)
        for num in nums:
            freq[num] += 1

        # Count of numbers divisible by g
        cnt = [0] * (max_num + 2)
        for g in range(1, max_num + 1):
            for multiple in range(g, max_num + 1, g):
                cnt[g] += freq[multiple]

        # Inclusion-Exclusion to find number of pairs with GCD exactly g
        total_pairs = [0] * (max_num + 2)
        for g in range(max_num, 0, -1):
            total_pairs[g] = cnt[g] * (cnt[g] - 1) // 2
            multiple = 2 * g
            while multiple <= max_num:
                total_pairs[g] -= total_pairs[multiple]
                multiple += g

        # Collect GCDs in sorted order with their frequencies
        sorted_gcds = []
        for g in range(1, max_num + 1):
            if total_pairs[g] > 0:
                sorted_gcds.append((g, total_pairs[g]))
        
        # Compute prefix sums to facilitate binary search
        prefix = []
        current = 0
        for g, cnt_g in sorted_gcds:
            current += cnt_g
            prefix.append((current, g))
        
        # Answer each query using binary search on prefix
        answer = []
        for q in queries:
            idx = q + 1  # Convert to 1-based index
            left, right = 0, len(prefix) - 1
            while left <= right:
                mid = (left + right) // 2
                if prefix[mid][0] >= idx:
                    right = mid - 1
                else:
                    left = mid + 1
            answer.append(prefix[left][1])
        return answer