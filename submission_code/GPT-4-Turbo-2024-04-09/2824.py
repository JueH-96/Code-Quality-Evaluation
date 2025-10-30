class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(n * 2) + str(n * 3)
        if '0' in concatenated:
            return False
        return len(set(concatenated)) == 9 and len(concatenated) == 9