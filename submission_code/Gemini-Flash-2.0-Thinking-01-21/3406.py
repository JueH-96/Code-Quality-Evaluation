class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        n_zero = zero
        n_one = one
        n = n_zero + n_one
        memo = {}
        MOD = 10**9 + 7
        
        def solve(index, zeros_count, last_digit, block_length):
            if index == n:
                return 1 if zeros_count == n_zero else 0
            if (index, zeros_count, last_digit, block_length) in memo:
                return memo[(index, zeros_count, last_digit, block_length)]
            
            count = 0
            # Try placing 0
            if zeros_count < n_zero:
                next_block_length_0 = block_length + 1 if last_digit == 0 else 1
                if next_block_length_0 <= limit:
                    count = (count + solve(index + 1, zeros_count + 1, 0, next_block_length_0)) % MOD
                    
            # Try placing 1
            ones_count = index - zeros_count
            if ones_count < n_one:
                next_block_length_1 = block_length + 1 if last_digit == 1 else 1
                if next_block_length_1 <= limit:
                    count = (count + solve(index + 1, zeros_count, 1, next_block_length_1)) % MOD
                    
            memo[(index, zeros_count, last_digit, block_length)] = count
            return count
            
        result = 0
        # Start with 0 if possible
        if n_zero > 0:
            result = (result + solve(1, 1, 0, 1)) % MOD
        # Start with 1 if possible
        if n_one > 0:
            result = (result + solve(1, 0, 1, 1)) % MOD
            
        return result