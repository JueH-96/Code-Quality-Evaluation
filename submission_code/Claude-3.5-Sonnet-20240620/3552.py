class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        def is_palindrome(num):
            return str(num) == str(num)[::-1]

        if n == 1:
            return str(max(i for i in range(9, 0, -1) if i % k == 0))

        upper_bound = 10**n - 1
        lower_bound = 10**(n-1)

        for num in range(upper_bound, lower_bound - 1, -1):
            left_half = str(num)[:n//2 + n%2]
            palindrome = int(left_half + left_half[-(n//2):][::-1])
            
            if palindrome <= upper_bound and palindrome % k == 0:
                return str(palindrome)

        return ""