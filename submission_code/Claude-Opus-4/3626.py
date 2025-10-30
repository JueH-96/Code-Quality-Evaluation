class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        while True:
            # Calculate the product of digits of n
            product = 1
            temp = n
            while temp > 0:
                product *= temp % 10
                temp //= 10
            
            # Check if product is divisible by t
            if product % t == 0:
                return n
            
            # Move to the next number
            n += 1