class Solution:
    def minLength(self, s: str) -> int:
        # Use a stack to simulate the removals.
        stack = []
        for char in s:
            stack.append(char)
            # Check for possible removals: "AB" or "CD".
            while len(stack) >= 2 and ((stack[-2] == 'A' and stack[-1] == 'B') or (stack[-2] == 'C' and stack[-1] == 'D')):
                # Remove the matching pair.
                stack.pop()  # pop the last character
                stack.pop()  # pop the second last character
        # The minimum possible length is the length of characters remaining.
        return len(stack)