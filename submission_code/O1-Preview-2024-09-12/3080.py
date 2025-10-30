class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        total_AND = nums[0]
        for num in nums[1:]:
            total_AND &= num

        subarray_count = 0
        current_AND = ~0
        for num in nums:
            current_AND &= num
            if current_AND == total_AND:
                subarray_count += 1
                current_AND = ~0
        
        return subarray_count if subarray_count > 0 else 1