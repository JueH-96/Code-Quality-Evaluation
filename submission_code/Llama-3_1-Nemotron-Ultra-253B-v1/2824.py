class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(2 * n) + str(3 * n)
        if len(concatenated) != 9:
            return False
        required_digits = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        return set(concatenated) == required_digits