class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(n * 2) + str(n * 3)
        return len(concatenated) == 9 and set(concatenated) == set("123456789")