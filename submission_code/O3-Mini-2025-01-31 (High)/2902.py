from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        groups = defaultdict(list)
        # Group numbers by their maximum digit
        for num in nums:
            # Compute the maximum digit by converting the number to a string
            max_digit = max(int(d) for d in str(num))
            groups[max_digit].append(num)
        
        max_sum_pair = -1
        # For each group with at least two numbers, get the two largest values
        for key in groups:
            if len(groups[key]) >= 2:
                groups[key].sort(reverse=True)
                current_sum = groups[key][0] + groups[key][1]
                max_sum_pair = max(max_sum_pair, current_sum)
        
        return max_sum_pair