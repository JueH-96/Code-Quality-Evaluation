class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            return str(max(i for i in range(9, 0, -1) if i % k == 0))
        
        max_num = int('9' * n)
        min_num = int('1' + '0' * (n - 1))
        
        for x in range(max_num, min_num - 1, -1):
            s = str(x)
            pal = int(s + s[-2::-1])
            if pal % k == 0:
                return str(pal)
        
        return ""