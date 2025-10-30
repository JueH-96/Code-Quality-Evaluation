class Solution:
    def kthCharacter(self, k: int) -> str:
        # Find the smallest n such that 2^n >= k
        n = 0
        while (1 << n) < k:
            n += 1
        
        def nextChar(c):
            return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
        
        def computeChar(n, k):
            if n == 0:
                return 'a'
            
            prev_length = 1 << (n-1)
            if k <= prev_length:
                return computeChar(n-1, k)
            else:
                prev_char = computeChar(n-1, k - prev_length)
                return nextChar(prev_char)
        
        return computeChar(n, k)