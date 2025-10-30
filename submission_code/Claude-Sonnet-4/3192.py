class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        # For each bit position from high to low, decide whether to flip it
        for i in range(n-1, -1, -1):
            # Current bit values in a and b
            bit_a = (a >> i) & 1
            bit_b = (b >> i) & 1
            
            # If we flip this bit (set x's bit i to 1):
            new_a = a ^ (1 << i)
            new_b = b ^ (1 << i)
            
            # Compare products: should we flip this bit?
            if new_a * new_b > a * b:
                a, b = new_a, new_b
        
        return (a * b) % MOD