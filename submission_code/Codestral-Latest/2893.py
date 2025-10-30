class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        even_max = 0
        odd_max = 0
        total_score = 0

        for num in nums:
            if num % 2 == 0:
                even_max = max(even_max, total_score + num)
            else:
                odd_max = max(odd_max, total_score + num)
            total_score += num

        return max(even_max, odd_max) + min(odd_max, even_max) - x