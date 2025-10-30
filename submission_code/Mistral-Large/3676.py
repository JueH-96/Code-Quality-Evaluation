class Solution:
    def smallestNumber(self, n: int) -> int:
        # Start from n and keep incrementing until we find a number with all set bits
        while True:
            # Convert the number to its binary representation and remove the '0b' prefix
            binary_representation = bin(n)[2:]
            # Check if all bits are set
            if '0' not in binary_representation:
                return n
            n += 1