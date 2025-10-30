class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        memo = {}
        
        def get_bit(num, i):
            return (num >> i) & 1
            
        def solve(index, current_x):
            if index < 0:
                val_a = a ^ current_x
                val_b = b ^ current_x
                return (val_a * val_b) % MOD
            
            if (index, current_x) in memo:
                return memo[(index, current_x)]
                
            bit_a = get_bit(a, index)
            bit_b = get_bit(b, index)
            
            if bit_a == bit_b:
                x_bit = 1 - bit_a
                result = solve(index - 1, current_x | (x_bit << index))
            else:
                res1 = solve(index - 1, current_x) # x_bit = 0
                res2 = solve(index - 1, current_x | (1 << index)) # x_bit = 1
                result = max(res1, res2)
                
            memo[(index, current_x)] = result
            return result
            
        return solve(n - 1, 0)