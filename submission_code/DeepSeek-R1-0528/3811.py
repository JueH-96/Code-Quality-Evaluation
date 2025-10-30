class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for i, char in enumerate(s):
            reversed_value = (ord('z') - ord(char)) + 1
            total += reversed_value * (i + 1)
        return total