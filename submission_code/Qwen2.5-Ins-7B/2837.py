class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for ops in range(61):
            target = num1 - ops * num2
            if target <= 0:
                break
            count_set_bits = bin(target).count('1')
            if count_set_bits <= ops and ops <= target:
                return ops
        return -1