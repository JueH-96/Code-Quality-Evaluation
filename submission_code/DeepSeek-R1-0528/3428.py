from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        count = Counter(nums)
        result = 0
        for num in count:
            if count[num] == 2:
                result ^= num
        return result