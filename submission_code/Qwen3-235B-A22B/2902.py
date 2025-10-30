from collections import defaultdict
from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_digits = defaultdict(list)
        for num in nums:
            max_d = max(int(c) for c in str(num))
            max_digits[max_d].append(num)
        
        max_sum = -1
        for group in max_digits.values():
            if len(group) >= 2:
                group.sort(reverse=True)
                current_sum = group[0] + group[1]
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum if max_sum != -1 else -1