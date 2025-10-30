class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num1 < num2:
            return -1
        for ans in range(1, 35):
            diff = num1 - ans * num2
            if diff <= 0:
                break
            cnt = 0
            for i in range(61):
                if diff & (1 << i):
                    cnt += 1
            if cnt <= ans and ans <= diff:
                return ans
        return -1