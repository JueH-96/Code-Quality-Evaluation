class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        for char in s:
            if char.isdigit():
                # Remove the closest non-digit character in stack (guaranteed to exist)
                stack.pop()
            else:
                stack.append(char)
        return "".join(stack)