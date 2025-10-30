class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        min_score = nums[0]
        for i in range(1, n):
            min_score &= nums[i]
        
        count = 0
        current_score = nums[0]
        for i in range(n):
            current_score &= nums[i]
            if current_score == min_score:
                count += 1
                if i < n - 1:
                    current_score = nums[i+1]
        return count