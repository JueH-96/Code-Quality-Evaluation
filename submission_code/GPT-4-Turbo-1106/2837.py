class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num2 == 0:
            return -1  # Impossible to subtract anything from num1
        
        operations = 0
        while num1 != 0:
            # If num2 is positive, we can only decrease num1 to 0 if it's already less than num2
            if num2 > 0 and num1 < num2:
                return -1
            
            # Find the largest power of 2 that is less than or equal to num1
            i = 0
            while (1 << i) <= num1:
                i += 1
            i -= 1
            
            # If num2 is negative, we can always make num1 0 by choosing i = 0
            if num2 < 0:
                return operations + num1
            
            # Subtract the value from num1
            num1 -= (1 << i) + num2
            operations += 1
            
            # If num1 went negative, it's impossible to reach 0
            if num1 < 0:
                return -1
        
        return operations