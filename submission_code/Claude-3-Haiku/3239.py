class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        operations = 0
        
        while x != y:
            if x > y:
                if x % 11 == 0:
                    x //= 11
                elif x % 5 == 0:
                    x //= 5
                else:
                    x -= 1
            else:
                if y % 11 == 0:
                    y //= 11
                elif y % 5 == 0:
                    y //= 5
                else:
                    y -= 1
            operations += 1
        
        return operations