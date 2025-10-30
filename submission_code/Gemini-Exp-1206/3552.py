class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            return str(k * (9 // k))
        
        half_len = (n + 1) // 2
        max_half = int('9' * half_len)
        
        for half in range(max_half, 10**(half_len - 1) - 1, -1):
            half_str = str(half)
            if n % 2 == 0:
                full_str = half_str + half_str[::-1]
            else:
                full_str = half_str + half_str[:-1][::-1]
            
            num = int(full_str)
            if num % k == 0:
                return full_str
        
        return ""