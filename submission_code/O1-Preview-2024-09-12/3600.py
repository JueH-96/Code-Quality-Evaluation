class Solution:
    def kthCharacter(self, k: int) -> str:
        def shift(c):
            return 'a' if c == 'z' else chr(ord(c) + 1)
        
        # Find n such that 2^n >= k
        n = 0
        while (1 << n) < k:
            n += 1
        
        stack = []
        while n > 0:
            L = 1 << (n - 1)
            if k > L:
                k -= L
                stack.append('shift')
            else:
                stack.append('same')
            n -=1
        
        c = 'a'
        while stack:
            action = stack.pop()
            if action == 'shift':
                c = shift(c)
            # 'same' action doesn't change c
        return c