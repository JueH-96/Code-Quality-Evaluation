class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(2 * n) + str(3 * n)
        if len(concatenated) != 9:
            return False
        sorted_concatenated = ''.join(sorted(concatenated))
        return sorted_concatenated == '123456789'