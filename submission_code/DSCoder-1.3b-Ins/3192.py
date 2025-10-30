class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        # XOR operation of all numbers from 0 to 2^n
        xor_all = 0
        for i in range(n):
            xor_all += (1 << i)
        
        # XOR operation of a and b
        xor_ab = a ^ b
        
        # Return the maximum value of (a XOR x) * (b XOR x)
        return max(xor_all & xor_ab, xor_all ^ xor_ab)