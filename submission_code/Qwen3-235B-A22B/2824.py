class Solution:
    def isFascinating(self, n: int) -> bool:
        concat_str = str(n) + str(2 * n) + str(3 * n)
        if len(concat_str) != 9:
            return False
        return sorted(concat_str) == ['1', '2', '3', '4', '5', '6', '7', '8', '9']