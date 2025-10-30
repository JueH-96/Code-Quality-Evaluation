from collections import Counter

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        window = Counter(nums[:k])
        max_sum = 0
        if len([v for v in window.values() if v == 1]) >= m:
            max_sum = sum(nums[:k])
        
        for i in range(k, len(nums)):
            window[nums[i]] += 1
            window[nums[i-k]] -= 1
            if window[nums[i-k]] == 0:
                del window[nums[i-k]]
            if len([v for v in window.values() if v == 1]) >= m:
                max_sum = max(max_sum, sum(nums[i-k+1:i+1]))
        
        return max_sum