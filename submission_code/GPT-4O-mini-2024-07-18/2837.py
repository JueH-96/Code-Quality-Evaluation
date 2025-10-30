class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # Calculate the maximum possible value we can subtract
        max_subtract = num1 + 2 * (60 * (60 + 1) // 2) + 60 * num2
        
        # If we can't reach zero, return -1
        if max_subtract < 0:
            return -1
        
        # The minimum number of operations needed
        operations = 0
        
        # We need to check if we can reach exactly zero
        for i in range(61):
            power_of_two = 1 << i  # 2^i
            subtract_value = power_of_two + num2
            
            # If subtracting this value from num1 is valid
            if num1 >= subtract_value:
                num1 -= subtract_value
                operations += 1
            
            # If we reach zero, return the number of operations
            if num1 == 0:
                return operations
        
        # If we exit the loop and haven't reached zero, return -1
        return -1