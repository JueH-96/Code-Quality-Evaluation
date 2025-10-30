from typing import List
import bisect

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def generate_palindromes(max_limit):
            palindromes = []
            # 1-digit to 9-digit palindromes
            for length in range(1, 10):
                half_len = (length + 1) // 2
                start = 10**(half_len -1) if half_len >1 else 1
                end = 10**half_len
                for first_half in range(start, end):
                    s = str(first_half)
                    if length %2 ==0:
                        full = s + s[::-1]
                    else:
                        full = s + s[:-1][::-1]
                    num = int(full)
                    if num < max_limit:
                        palindromes.append(num)
            return palindromes
        
        palindromes = generate_palindromes(10**9)
        palindromes.sort()
        
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        prefix_sum = [0] * (n +1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums_sorted[i]
        total_sum = prefix_sum[-1]
        
        min_cost = float('inf')
        for y in palindromes:
            i = bisect.bisect_left(nums_sorted, y)
            sum_left = prefix_sum[i]
            len_left = i
            sum_right = total_sum - sum_left
            len_right = n - i
            cost = y * len_left - sum_left + sum_right - y * len_right
            if cost < min_cost:
                min_cost = cost
        return min_cost