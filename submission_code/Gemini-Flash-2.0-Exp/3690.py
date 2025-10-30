class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        ans = n
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                sub = s[i:i + length]
                
                zeros = 0
                for char in sub:
                    if char == '0':
                        zeros += 1
                
                ones = length - zeros
                
                if zeros <= numOps or ones <= numOps:
                    ans = min(ans, length)
        return ans