class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(2 * n) + str(3 * n)
        return len(concatenated) == 9 and '0' not in concatenated and len(set(concatenated)) == 9