from typing import List
from collections import Counter

MOD = 10**9 + 7

def comb(n: int, k: int) -> int:
    if k < 0 or n < k:
        return 0
    if k == 0:
        return 1
    elif k == 1:
        return n
    elif k == 2:
        return n * (n - 1) // 2
    return 0

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            x = nums[i]
            left_part = nums[:i]
            right_part = nums[i+1:]
            
            left_x_count = left_part.count(x)
            left_non_x = [num for num in left_part if num != x]
            left_non_x_count = len(left_non_x)
            left_non_x_freq = Counter(left_non_x)
            
            right_x_count = right_part.count(x)
            right_non_x = [num for num in right_part if num != x]
            right_non_x_count = len(right_non_x)
            right_non_x_freq = Counter(right_non_x)
            
            # Compute case1
            left_count = i
            right_count = n - i - 1
            total_ways_case1 = comb(left_count, 2) * comb(right_count, 2)
            
            ways_left_0 = comb(left_x_count, 0) * comb(left_non_x_count, 2)
            ways_left_1 = comb(left_x_count, 1) * comb(left_non_x_count, 1)
            ways_right_0 = comb(right_x_count, 0) * comb(right_non_x_count, 2)
            ways_right_1 = comb(right_x_count, 1) * comb(right_non_x_count, 1)
            
            subtract = (ways_left_0 * ways_right_0 + ways_left_0 * ways_right_1 + ways_left_1 * ways_right_0)
            case1 = (total_ways_case1 - subtract) % MOD
            
            # Compute case2
            case2 = 0
            
            # Subcase 2.1: a=0, b=1
            left_valid_0 = comb(left_non_x_count, 2)
            sum_c = 0
            for c in left_non_x_freq.values():
                sum_c += c * (c - 1) // 2
            left_valid_0 = (left_valid_0 - sum_c) % MOD
            
            ways_right_x = comb(right_x_count, 1)
            sum_a = 0
            for a in left_non_x_freq:
                cnt_a_in_right = right_non_x_freq.get(a, 0)
                sum_a += cnt_a_in_right * left_non_x_freq[a] * (left_non_x_count - left_non_x_freq[a])
            valid_non_x_count = (left_valid_0 * right_non_x_count - sum_a) % MOD
            ways_subcase_2_1 = (ways_right_x * valid_non_x_count) % MOD
            case2 = (case2 + ways_subcase_2_1) % MOD
            
            # Subcase 2.2: a=1, b=0
            left_ways = comb(left_x_count, 1)
            sum_right_ways = 0
            for l in left_non_x_freq:
                cnt_l_in_right = right_non_x_freq.get(l, 0)
                valid_right_non_x = right_non_x_count - cnt_l_in_right
                sum_right_ways = (sum_right_ways + left_non_x_freq[l] * comb(valid_right_non_x, 2)) % MOD
            ways_subcase_2_2 = (left_ways * sum_right_ways) % MOD
            case2 = (case2 + ways_subcase_2_2) % MOD
            
            total = (total + case1 + case2) % MOD
        
        return total % MOD