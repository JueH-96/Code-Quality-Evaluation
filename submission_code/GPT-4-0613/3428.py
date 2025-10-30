class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import Counter
        duplicates = [k for k, v in Counter(nums).items() if v > 1]
        result = 0
        for num in duplicates:
            result ^= num
        return result