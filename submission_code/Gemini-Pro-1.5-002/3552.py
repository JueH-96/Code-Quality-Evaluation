class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            return str(k * (9 // k))

        if n % 2 == 0:
            prefix = "9" * (n // 2)
            for i in range(int(prefix), -1, -1):
                s_prefix = str(i)
                if len(s_prefix) < n // 2:
                    s_prefix = "0" * (n // 2 - len(s_prefix)) + s_prefix

                palindrome = s_prefix + s_prefix[::-1]
                num = int(palindrome)
                if num % k == 0:
                    return palindrome
        else:
            prefix = "9" * (n // 2)
            for i in range(int(prefix), -1, -1):
                s_prefix = str(i)
                if len(s_prefix) < n // 2:
                    s_prefix = "0" * (n // 2 - len(s_prefix)) + s_prefix
                for j in range(9, -1, -1):
                    palindrome = s_prefix + str(j) + s_prefix[::-1]
                    num = int(palindrome)

                    if num % k == 0:
                        return palindrome
        return ""