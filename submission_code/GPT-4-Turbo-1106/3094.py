from collections import Counter

class Solution:
    def minOperations(self, nums: list[int]) -> int:
        count = Counter(nums)
        operations = 0
        
        for key in count:
            if count[key] % 2 != 0:
                count[key] -= 3
                operations += 1
            if count[key] < 0:
                return -1
        
        return operations + sum(count.values()) // 2