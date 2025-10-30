class Solution:
    def smallestNumber(self, n: int) -> int:
        x = n
        while True:
            # Convert x to binary string and remove the '0b' prefix
            binary = bin(x)[2:]
            # Check if all bits are set (all '1's)
            if all(bit == '1' for bit in binary):
                return x
            x += 1