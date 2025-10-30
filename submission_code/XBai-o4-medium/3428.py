class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        count = {}
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
        duplicates = [num for num in count if count[num] == 2]
        res = 0
        for num in duplicates:
            res ^= num
        return res