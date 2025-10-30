class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num1 < num2:
            return -1
        if num1 == num2 and num2 == 0:
            return 0
        if num2 >= 0 and num1 < num2:
            return -1
        if num2 >= 0 and num1 == num2:
            return -1
        
        for k in range(1, 61):
            diff = num1 - k * num2
            if diff < k:
                continue
            
            count = 0
            temp = diff
            while temp > 0:
                if temp & 1:
                    count += 1
                temp >>= 1
            if count <= k:
                return k
        return -1