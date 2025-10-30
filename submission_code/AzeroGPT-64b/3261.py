class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        data = [defaultdict(int), defaultdict(int)]
        for number in nums:
            binData = bin(number)[2:]
            for index, c in enumerate(binData):
                data[0][index + len(binData) - len(binData)] += int(c)
                data[1][index + len(binData) - len(binData)] += 1 if number != (1 << (len(binData) - index - 1)) else 0

        result = 0
        for index in range(29, -1, -1):
            if data[0][index] == 0:
                continue
            if k > 0 and data[0][index] < data[1][index]:
                k -= 1
            else:
                result |= 1 << index
        return result