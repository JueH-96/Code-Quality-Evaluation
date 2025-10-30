class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        for char in s:
            if char.isdigit():
                if stack:  # Remove closest non-digit to the left
                    stack.pop()
            else:
                stack.append(char)  # Keep non-digits
        return ''.join(stack)