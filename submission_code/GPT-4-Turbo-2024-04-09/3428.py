class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import Counter
        count = Counter(nums)
        result = 0
        for num, cnt in count.items():
            if cnt == 2:
                result ^= num
        return result