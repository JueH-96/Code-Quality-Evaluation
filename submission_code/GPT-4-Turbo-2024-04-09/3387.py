class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = (n - 1) // 2
        
        if n % 2 == 1:
            # Odd length, single median
            return abs(nums[median_index] - k)
        else:
            # Even length, take the larger of the two middle values
            return abs(nums[median_index + 1] - k)