class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        best_even = -10**18
        best_odd = -10**18
        
        if nums[0] % 2 == 0:
            best_even = nums[0]
        else:
            best_odd = nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] % 2 == 0:
                candidate = max(best_even + nums[i], best_odd + nums[i] - x)
                best_even = max(best_even, candidate)
            else:
                candidate = max(best_odd + nums[i], best_even + nums[i] - x)
                best_odd = max(best_odd, candidate)
        
        return max(best_even, best_odd)