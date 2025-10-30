class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        score = 0
        for num in nums:
            score &= num
            if score == 0:
                break
        if score != 0:
            return 1
        count = 0
        current_score = 0xFFFFFFFF
        for num in nums:
            current_score &= num
            if current_score == 0:
                count += 1
                current_score = 0xFFFFFFFF
        return count