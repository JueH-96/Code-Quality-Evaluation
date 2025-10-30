class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        def digit_diff(a, b):
            a_str, b_str = str(a), str(b)
            return sum(1 for x, y in zip(a_str, b_str) if x != y)
        
        total_diff = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                total_diff += digit_diff(nums[i], nums[j])
        
        return total_diff