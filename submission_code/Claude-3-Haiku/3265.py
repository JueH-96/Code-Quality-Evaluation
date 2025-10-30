class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = 0
        
        for i in range(n):
            for j in range(i, n):
                if abs(nums[i] - nums[j]) == k:
                    curr_sum = sum(nums[i:j+1])
                    max_sum = max(max_sum, curr_sum)
        
        return max_sum