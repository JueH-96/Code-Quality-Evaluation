class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        char_to_index = {chr(ord('a') + i): i for i in range(26)}
        
        total_length = 0
        
        for char in s:
            if char == 'z':
                total_length = (total_length + 2) % MOD
            else:
                c = char_to_index[char]
                z_shift = (25 - c) % 26
                if z_shift <= t:
                    total_length = (total_length + 2) % MOD
                else:
                    total_length = (total_length + 1) % MOD
        
        return total_length