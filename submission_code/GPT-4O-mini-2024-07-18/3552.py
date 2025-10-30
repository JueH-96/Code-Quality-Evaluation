class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, -1, -1):
                if i % k == 0:
                    return str(i)

        half_length = (n + 1) // 2
        start = 10**half_length - 1
        end = 10**(half_length - 1)

        for i in range(start, end - 1, -1):
            palindrome = str(i) + str(i)[-2::-1] if n % 2 == 0 else str(i) + str(i)[-1::-1]
            if int(palindrome) % k == 0:
                return palindrome
        
        return ""