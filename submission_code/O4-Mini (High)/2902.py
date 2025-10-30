from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Group numbers by their maximum digit
        groups = {}
        for num in nums:
            max_digit = int(max(str(num)))
            groups.setdefault(max_digit, []).append(num)
        
        # For each group with at least two numbers, take the two largest
        ans = -1
        for group in groups.values():
            if len(group) >= 2:
                group.sort(reverse=True)
                ans = max(ans, group[0] + group[1])
        
        return ans