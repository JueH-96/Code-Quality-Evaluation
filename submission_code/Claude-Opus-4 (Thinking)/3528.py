class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        score = 0
        current_pos = 0
        current_val = nums[0]
        
        for i in range(1, n):
            # Jump if we've reached the end or found a better/equal value
            if i == n - 1 or nums[i] >= current_val:
                score += (i - current_pos) * current_val
                current_pos = i
                current_val = nums[i]
        
        return score