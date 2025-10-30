class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        operations = 0
        
        while x != y:
            if x > y:
                if x % 11 == 0 and x // 11 >= y:
                    x //= 11
                elif x % 5 == 0 and x // 5 >= y:
                    x //= 5
                else:
                    x -= 1
            else:
                x += 1
            operations += 1
        
        return operations