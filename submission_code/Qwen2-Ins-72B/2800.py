class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        
        for char in s:
            if stack and stack[-1] + char in ("AB", "CD"):
                stack.pop()
            else:
                stack.append(char)
                
        return len(stack)