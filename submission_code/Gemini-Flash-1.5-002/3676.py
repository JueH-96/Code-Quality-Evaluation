class Solution:
    def smallestNumber(self, n: int) -> int:
        x = n
        while True:
            binary = bin(x)[2:]
            if all(bit == '1' for bit in binary):
                return x
            x += 1