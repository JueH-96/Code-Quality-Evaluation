class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        
        max_sum = 0
        current_sum = sum(nums[:k])
        distinct_count = len(set(nums[:k]))
        
        if distinct_count >= m:
            max_sum = current_sum
        
        for i in range(k, n):
            current_sum += nums[i] - nums[i - k]
            # Update the distinct count
            window = nums[i - k + 1:i + 1]
            distinct_count = len(set(window))
            
            if distinct_count >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum