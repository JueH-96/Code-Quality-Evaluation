class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1 and k == 1:
            return "9"
        if n == 1 and k > 1:
            if k == 2:
                return "8"
            elif k == 3:
                return "9"
            elif k == 4:
                return "8"
            elif k == 5:
                return "5"
            elif k == 6:
                return "6"
            elif k == 7:
                return "7"
            elif k == 8:
                return "8"
            elif k == 9:
                return "9"

        upper_bound = 10**n -1
        lower_bound = 10**(n-1)
        
        for i in range(upper_bound, lower_bound -1, -1):
            s = str(i)
            palindrome = s + s[::-1]
            num = int(palindrome)
            if num % k == 0:
                return str(num)
        return ""