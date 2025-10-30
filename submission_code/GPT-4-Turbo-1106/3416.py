class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        def digit_diff(a, b):
            return sum(d1 != d2 for d1, d2 in zip(str(a), str(b)))
        
        total_diff = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                total_diff += digit_diff(nums[i], nums[j])
        
        return total_diff