class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        from collections import Counter
        
        n = len(nums)
        if n < k:
            return 0
        
        count = Counter(nums[:k])
        current_sum = sum(nums[:k])
        max_sum = 0
        
        if len(count) >= m:
            max_sum = current_sum
        
        for i in range(k, n):
            current_sum += nums[i] - nums[i - k]
            count[nums[i]] += 1
            count[nums[i - k]] -= 1
            if count[nums[i - k]] == 0:
                del count[nums[i - k]]
            if len(count) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum