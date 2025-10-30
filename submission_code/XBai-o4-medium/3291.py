from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        if not nums:
            return True
        
        # Group the numbers based on consecutive same bit counts
        groups = []
        current_group = [nums[0]]
        current_bit = bin(nums[0]).count('1')
        
        for num in nums[1:]:
            bit = bin(num).count('1')
            if bit == current_bit:
                current_group.append(num)
            else:
                groups.append(current_group)
                current_group = [num]
                current_bit = bit
        groups.append(current_group)
        
        # Create the sorted version of each group
        sorted_groups = []
        for group in groups:
            sorted_groups.extend(sorted(group))
        
        # Check if the combined sorted groups are non-decreasing
        for i in range(len(sorted_groups) - 1):
            if sorted_groups[i] > sorted_groups[i+1]:
                return False
        return True