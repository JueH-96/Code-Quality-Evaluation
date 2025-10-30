class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_sum = 0
        
        for i in range(n):
            min_val = nums[i]
            max_val = nums[i]
            for j in range(i, min(i + k, n)):
                min_val = min(min_val, nums[j])
                max_val = max(max_val, nums[j])
                total_sum += min_val + max_val
        
        return total_sum