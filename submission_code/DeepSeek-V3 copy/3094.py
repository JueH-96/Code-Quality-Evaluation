from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        operations = 0
        for num, cnt in count.items():
            if cnt == 1:
                return -1
            # Try to use as many 3-operations as possible
            operations += cnt // 3
            remainder = cnt % 3
            if remainder == 1:
                # Need to convert one 3-operation into two 2-operations
                operations -= 1
                operations += 2
            elif remainder == 2:
                operations += 1
        return operations