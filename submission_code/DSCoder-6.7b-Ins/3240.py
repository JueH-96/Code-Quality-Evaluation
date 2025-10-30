class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        num = 0
        total = 0
        while total + bin(num).count('1') <= k:
            total += bin(num)[2:].count('1')
            num += 1
        return num - 1