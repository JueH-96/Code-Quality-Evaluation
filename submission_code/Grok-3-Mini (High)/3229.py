import bisect
from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Sort the nums array
        sorted_nums = sorted(nums)
        n = len(nums)
        
        # Compute cumulative sum
        cumsum = [0]
        for num in sorted_nums:
            cumsum.append(cumsum[-1] + num)
        total_sum_val = cumsum[-1]
        
        # Generate all palindromic numbers less than 10^9
        palindromes = []
        for len_d in range(1, 10):  # Number of digits from 1 to 9
            num_half_digits = (len_d + 1) // 2
            if num_half_digits > 1:
                start = 10 ** (num_half_digits - 1)
            else:
                start = 1
            end = 10 ** num_half_digits
            for first_part_int in range(start, end):
                s_first = str(first_part_int)
                floor_d_half = len_d // 2
                mirror_part_str = s_first[:floor_d_half][::-1]  # Reverse the first floor(d/2) characters
                full_palin_str = s_first + mirror_part_str
                pal_num = int(full_palin_str)
                palindromes.append(pal_num)
        
        # Now, for each palindromic y, compute the cost and find the minimum
        min_cost = float('inf')
        for y in palindromes:
            cnt_le = bisect.bisect_right(sorted_nums, y)  # Number of elements <= y
            sum_le = cumsum[cnt_le]  # Sum of elements <= y
            cost = 2 * y * cnt_le - 2 * sum_le + total_sum_val - y * n
            if cost < min_cost:
                min_cost = cost
        
        return int(min_cost)