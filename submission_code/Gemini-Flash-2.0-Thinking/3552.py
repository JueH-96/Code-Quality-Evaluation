class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, 0, -1):
                if i % k == 0:
                    return str(i)
            return ""

        if n % 2 == 0:
            half_len = n // 2
            for i in range(10**half_len - 1, 10**(half_len - 1) - 1, -1):
                s = str(i)
                palindrome = s + s[::-1]
                if int(palindrome) % k == 0:
                    return palindrome
        else:
            half_len = (n + 1) // 2
            for i in range(10**half_len - 1, 10**(half_len - 1) - 1, -1):
                s = str(i)
                palindrome = s + s[:-1][::-1]
                if int(palindrome) % k == 0:
                    return palindrome

        return ""