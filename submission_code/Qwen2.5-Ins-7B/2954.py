from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        current_sum = 0
        count = defaultdict(int)
        
        for i in range(len(nums)):
            current_sum += nums[i]
            count[nums[i]] += 1
            
            if i >= k:
                current_sum -= nums[i - k]
                count[nums[i - k]] -= 1
                if count[nums[i - k]] == 0:
                    del count[nums[i - k]]
            
            if i >= k - 1 and len(count) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum