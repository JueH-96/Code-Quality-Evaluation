class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        result = 0
        power = 1
        
        for i in range(len(s) - 1, -1, -1):
            char = s[i]
            if char == 'a':
                result = (result + 2 * power - 1) % MOD
                power = 2 * power % MOD
            else:
                diff_a = ord('a') - ord('z')
                char_diff = ord(char) - ord('a')
                result = (result + ((1 + char_diff + t) * power - 1) % MOD) % MOD
                #Now let's compute power modulo MOD
                power = ((t + char_diff + 1) * power) % MOD
            power = power * pow((ord(char) - diff_a) % MOD, t, MOD) % MOD
        
        return result