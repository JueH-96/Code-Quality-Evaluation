from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        groups = defaultdict(list)
        for num in nums:
            max_d = max(int(d) for d in str(num))
            groups[max_d].append(num)
        
        max_sum = -1
        for group in groups.values():
            if len(group) >= 2:
                sorted_group = sorted(group, reverse=True)
                current_sum = sorted_group[0] + sorted_group[1]
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum if max_sum != -1 else -1