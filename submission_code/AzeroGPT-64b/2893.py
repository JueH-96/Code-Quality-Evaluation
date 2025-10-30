class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        score, score_to_odd, score_to_even = nums[0], 0, nums[0]
        
        for i in range(1, n):
            num = nums[i]
            score_to_even = score_to_even + num - x if num % 2 == 0 else max(score_to_even, score_to_odd + num)
            score_to_odd = score_to_odd + num - x if num % 2 == 1 else max(score_to_odd, score_to_even + num)
            
        return max(score_to_odd, score_to_even)