class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        min_score = float('inf')
        current_score = nums[0]
        subarray_count = 1

        for num in nums[1:]:
            current_score &= num
            if current_score == 0:
                subarray_count += 1
                current_score = num

        return subarray_count