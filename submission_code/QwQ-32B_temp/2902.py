from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def get_max_digit(n):
            max_d = 0
            while n > 0:
                digit = n % 10
                if digit > max_d:
                    max_d = digit
                n = n // 10
            return max_d
        
        groups = defaultdict(list)
        for num in nums:
            md = get_max_digit(num)
            groups[md].append(num)
        
        max_sum = -1
        for group in groups.values():
            if len(group) >= 2:
                sorted_group = sorted(group, reverse=True)
                current_sum = sorted_group[0] + sorted_group[1]
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum