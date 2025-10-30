from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq_map = Counter(nums)
        operations = 0
        for count in freq_map.values():
            if count == 1:
                return -1
            if count % 3 == 0:
                operations += count // 3
            else:
                operations += (count // 3) + 1
        return operations