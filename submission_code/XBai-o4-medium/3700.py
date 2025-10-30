import math
from typing import List
from collections import Counter

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute prefix and suffix counters
        prefix = [Counter() for _ in range(n + 1)]
        for i in range(n):
            prefix[i + 1] = prefix[i].copy()
            prefix[i + 1][nums[i]] += 1
        
        suffix = [Counter() for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1].copy()
            suffix[i][nums[i]] += 1
        
        def comb(n, k):
            if n < 0 or k < 0:
                return 0
            if n < k:
                return 0
            try:
                return math.comb(n, k)
            except:
                return 0
        
        total = 0
        
        for i in range(n):
            x = nums[i]
            left_count = prefix[i][x]
            right_count = suffix[i + 1][x]
            left_non_x_count = i - left_count
            right_non_x_count = (n - i - 1) - right_count
            
            # Process m=3
            m3 = 0
            for a, b in [(0, 2), (1, 1), (2, 0)]:
                left_ways = comb(left_count, a) * comb(left_non_x_count, 2 - a)
                right_ways = comb(right_count, b) * comb(right_non_x_count, 2 - b)
                m3 += left_ways * right_ways
            total = (total + m3) % MOD
            
            # Process m=4
            m4 = 0
            for a, b in [(1, 2), (2, 1)]:
                left_ways = comb(left_count, a) * comb(left_non_x_count, 2 - a)
                right_ways = comb(right_count, b) * comb(right_non_x_count, 2 - b)
                m4 += left_ways * right_ways
            total = (total + m4) % MOD
            
            # Process m=5
            if left_count >= 2 and right_count >= 2:
                m5 = comb(left_count, 2) * comb(right_count, 2)
                total = (total + m5) % MOD
            
            # Process m=2
            # Case1: a=0, b=1
            case1 = 0
            if right_count >= 1 and left_non_x_count >= 2:
                total_ways_case1 = comb(left_non_x_count, 2) * comb(right_count, 1) * right_non_x_count
                sum_overlapping = 0
                all_ys = set(prefix[i].keys()).union(suffix[i + 1].keys())
                for y in all_ys:
                    if y == x:
                        continue
                    left_freq_y = prefix[i].get(y, 0)
                    right_freq_y = suffix[i + 1].get(y, 0)
                    if left_freq_y == 0 or right_freq_y == 0:
                        continue
                    term1 = comb(left_freq_y, 1) * comb(left_non_x_count - left_freq_y, 1)
                    term2 = comb(left_freq_y, 2)
                    ways_left_has_y = term1 + term2
                    ways_right_is_y = comb(right_count, 1) * right_freq_y
                    sum_overlapping += ways_left_has_y * ways_right_is_y
                valid_case1 = (total_ways_case1 - sum_overlapping)
                case1 = valid_case1
            else:
                case1 = 0
            
            # Case2: a=1, b=0
            case2 = 0
            if left_count >= 1 and right_non_x_count >= 2:
                total_ways_case2 = comb(left_count, 1) * left_non_x_count * comb(right_non_x_count, 2)
                sum_overlapping_2 = 0
                all_ys = set(prefix[i].keys()).union(suffix[i + 1].keys())
                for y in all_ys:
                    if y == x:
                        continue
                    left_freq_y = prefix[i].get(y, 0)
                    right_freq_y = suffix[i + 1].get(y, 0)
                    if left_freq_y == 0 or right_freq_y == 0:
                        continue
                    term1 = comb(right_freq_y, 1) * comb(right_non_x_count - right_freq_y, 1)
                    term2 = comb(right_freq_y, 2)
                    ways_right_has_y = term1 + term2
                    ways_left_is_y = comb(left_count, 1) * left_freq_y
                    sum_overlapping_2 += ways_left_is_y * ways_right_has_y
                valid_case2 = (total_ways_case2 - sum_overlapping_2)
                case2 = valid_case2
            else:
                case2 = 0
            
            total = (total + case1 + case2) % MOD
        
        return total