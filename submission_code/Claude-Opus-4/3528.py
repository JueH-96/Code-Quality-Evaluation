class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        score = 0
        current_pos = 0
        current_val = nums[0]
        
        for i in range(1, n):
            # If we find a position with a larger value or reach the end
            if nums[i] > current_val or i == n - 1:
                # Jump from current_pos to i
                score += (i - current_pos) * current_val
                current_pos = i
                current_val = nums[i]
        
        return score