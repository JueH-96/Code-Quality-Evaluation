class Solution:
    def smallestNumber(self, n: int) -> int:
        # Start checking from n
        x = n
        
        while True:
            # Convert number to binary string and remove '0b' prefix
            binary = bin(x)[2:]
            
            # Check if binary representation contains only 1s
            if '0' not in binary:
                return x
            
            # If not, move to next number
            x += 1