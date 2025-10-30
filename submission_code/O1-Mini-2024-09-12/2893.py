class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        INF = float('-inf')
        if nums[0] % 2 == 0:
            score_even = nums[0]
            score_odd = INF
        else:
            score_odd = nums[0]
            score_even = INF
        
        for num in nums[1:]:
            if num % 2 == 0:
                new_score_even = max(score_even + num, score_odd + num - x)
                new_score_odd = score_odd
            else:
                new_score_odd = max(score_odd + num, score_even + num - x)
                new_score_even = score_even
            score_even, score_odd = new_score_even, new_score_odd
        
        return max(score_even, score_odd)