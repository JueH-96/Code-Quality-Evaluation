class Solution:
    def maxOperations(self, s: str) -> int:
        ans = 0
        stack = []
        for i, c in enumerate(s):
            if i > 0 and c == '1' and s[i-1] == '0':
                if stack and stack[-1] == 1:
                    ans += 1
                    stack.pop()
                else:
                    stack.append(0)
            elif c == '1': 
                stack.append(1)
                
        return ans