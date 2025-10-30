class Solution:
    def smallestNumber(self, n: int) -> int:
        binary_n = bin(n)[2:]
        num_bits = len(binary_n)
        target = (1 << num_bits) - 1
        if target >= n:
            return target
        else:
            return (1 << (num_bits + 1)) - 1