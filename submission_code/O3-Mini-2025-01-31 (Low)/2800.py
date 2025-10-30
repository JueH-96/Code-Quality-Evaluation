class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for char in s:
            if stack:
                # Check if last two characters form "AB" or "CD"
                if stack[-1] == 'A' and char == 'B':
                    stack.pop()
                    continue
                if stack[-1] == 'C' and char == 'D':
                    stack.pop()
                    continue
            stack.append(char)
        return len(stack)
        
# Example usage:
# sol = Solution()
# print(sol.minLength("ABFCACDB"))  # Output: 2
# print(sol.minLength("ACBBD"))     # Output: 5