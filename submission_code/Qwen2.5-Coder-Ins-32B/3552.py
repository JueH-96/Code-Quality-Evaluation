class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, 0, -1):
                if i % k == 0:
                    return str(i)
            return "0"
        
        half = (n + 1) // 2
        start = int('9' * half)
        
        for i in range(start, -1, -1):
            s = str(i)
            if n % 2 == 0:
                candidate = s + s[::-1]
            else:
                candidate = s + s[-2::-1]
            
            num = int(candidate)
            if num % k == 0:
                return candidate
        
        return "0"