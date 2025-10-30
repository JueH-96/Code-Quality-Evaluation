from collections import defaultdict

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
        xor_result = 0
        for num in count:
            if count[num] == 2:
                xor_result ^= num
        return xor_result