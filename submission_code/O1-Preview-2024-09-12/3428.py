class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import Counter
        counts = Counter(nums)
        duplicates = [num for num, count in counts.items() if count == 2]
        result = 0
        for num in duplicates:
            result ^= num
        return result