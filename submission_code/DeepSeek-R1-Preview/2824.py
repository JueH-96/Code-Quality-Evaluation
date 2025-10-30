class Solution:
    def isFascinating(self, n: int) -> bool:
        s = str(n) + str(2 * n) + str(3 * n)
        if len(s) != 9:
            return False
        required = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        return set(s) == required