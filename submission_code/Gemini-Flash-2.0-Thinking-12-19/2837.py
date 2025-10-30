class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num1 == 0:
            return 0
        for k in range(1, 61):
            target = num1 - k * num2
            if target < 0:
                return -1
            if target == 0:
                if k == 0:
                    return 0
                else:
                    continue
            set_bits = 0
            temp_target = target
            while temp_target > 0:
                if temp_target % 2 == 1:
                    set_bits += 1
                temp_target //= 2
            if set_bits <= k:
                return k
        return -1