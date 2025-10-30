class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, -1, -1):
                if i % k == 0:
                    return str(i)
            return ""
        
        prefix_len = (n + 1) // 2
        for prefix_val in range(10**prefix_len - 1, -1, -1):
            prefix_str = str(prefix_val)
            if n % 2 == 1:
                palindrome_str = prefix_str + prefix_str[:-1][::-1]
            else:
                palindrome_str = prefix_str + prefix_str[::-1]
            if len(palindrome_str) != n:
                continue
            palindrome_int = int(palindrome_str)
            if palindrome_int % k == 0:
                return palindrome_str
        return ""