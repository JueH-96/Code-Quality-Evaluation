import math

class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 1000000007
        
        # Compute frequency of each character in s
        cnt_list = [0] * 26
        for char in s:
            idx = ord(char) - ord('a')
            cnt_list[idx] += 1
        
        # Compute dp_z where dp_z[k] is the length after k transformations starting from 'z' (index 25)
        dp_z = [0] * (t + 1)
        dp_z[0] = 1
        for curr_t in range(1, t + 1):
            prev = curr_t - 1
            a = 1 if prev <= 25 else dp_z[prev - 25]
            b = 1 if prev <= 24 else dp_z[prev - 24]
            dp_z[curr_t] = (a + b) % MOD
        
        # Compute the total length modulo MOD
        sum_mod = 0
        for i in range(26):
            if t <= 25 - i:
                contrib_length = 1
            else:
                s_val = t - 25 + i
                contrib_length = dp_z[s_val]
            prod = (cnt_list[i] * contrib_length) % MOD
            sum_mod = (sum_mod + prod) % MOD
        
        return sum_mod