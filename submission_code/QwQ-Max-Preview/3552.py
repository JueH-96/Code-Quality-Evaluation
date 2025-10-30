class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        m = (n + 1) // 2
        current_part = '9' * m
        
        def generate_palindrome(fp):
            if n % 2 == 0:
                return fp + fp[::-1]
            else:
                return fp + fp[:-1][::-1]
        
        def is_divisible(s):
            remainder = 0
            for c in s:
                remainder = (remainder * 10 + int(c)) % k
            return remainder == 0
        
        while True:
            if current_part[0] == '0':
                break
            palindrome = generate_palindrome(current_part)
            if is_divisible(palindrome):
                return palindrome
            # Decrement current_part
            new_part = list(current_part)
            i = len(new_part) - 1
            while i >= 0 and new_part[i] == '0':
                new_part[i] = '9'
                i -= 1
            if i == -1:
                break  # All zeros, no more possible
            new_part[i] = str(int(new_part[i]) - 1)
            new_part_str = ''.join(new_part)
            if new_part_str[0] == '0':
                break
            current_part = new_part_str
        
        return ""