from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        if not nums:
            return True
        
        # Calculate the number of set bits for each number
        bits = [bin(num).count('1') for num in nums]
        
        # Split into regions where each region has the same number of set bits consecutively
        regions = []
        current_region = [nums[0]]
        prev_bit = bits[0]
        
        for i in range(1, len(nums)):
            current_bit = bits[i]
            if current_bit == prev_bit:
                current_region.append(nums[i])
            else:
                regions.append(current_region)
                current_region = [nums[i]]
                prev_bit = current_bit
        regions.append(current_region)  # Add the last region
        
        # Sort each region
        sorted_regions = [sorted(region) for region in regions]
        
        # Build the candidate array by concatenating sorted regions
        candidate = []
        for r in sorted_regions:
            candidate.extend(r)
        
        # Check if the candidate array is non-decreasing
        for i in range(1, len(candidate)):
            if candidate[i] < candidate[i-1]:
                return False
        return True