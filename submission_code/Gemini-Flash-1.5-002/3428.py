class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        seen = set()
        duplicates = set()
        for num in nums:
            if num in seen:
                duplicates.add(num)
            else:
                seen.add(num)
        result = 0
        for num in duplicates:
            result ^= num
        return result