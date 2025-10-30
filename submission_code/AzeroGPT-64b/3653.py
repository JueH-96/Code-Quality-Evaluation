class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n_sum, pre_sum = sum(nums), 0
        if len(nums) % k != 0:
            raise ValueError("The length of nums is not divisible by k.")
        
        min_sums = [float('inf')] * k
        min_sums[0] = 0
        max_sum = min_sums[0]
        
        for i in range(1, len(nums) + 1):
            sub_len = i % k
            min_sums[sub_len] = min(min_sums[sub_len], pre_sum) 
          
            if i >= k:
                max_sum = max(max_sum, pre_sum - min_sums[i % k])
            pre_sum += nums[i - 1]
        return max_sum if max_sum >= 0 else -n_sum