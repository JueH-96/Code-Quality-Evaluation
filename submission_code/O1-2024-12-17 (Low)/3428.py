class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import Counter
        count_map = Counter(nums)
        result = 0
        for num, freq in count_map.items():
            if freq == 2:
                result ^= num
        return result