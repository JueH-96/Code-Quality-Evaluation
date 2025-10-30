from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_digit_groups = {}
        
        for num in nums:
            s = str(num)
            max_d = max(int(c) for c in s)
            if max_d not in max_digit_groups:
                max_digit_groups[max_d] = []
            max_digit_groups[max_d].append(num)
        
        max_total = -1
        
        for group in max_digit_groups.values():
            if len(group) >= 2:
                sorted_group = sorted(group, reverse=True)
                current_sum = sorted_group[0] + sorted_group[1]
                if current_sum > max_total:
                    max_total = current_sum
        
        return max_total if max_total != -1 else -1