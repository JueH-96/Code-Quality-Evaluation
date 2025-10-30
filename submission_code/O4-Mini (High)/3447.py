class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        for ch in s:
            if ch.isdigit():
                # Remove the closest non-digit character to the left
                stack.pop()
            else:
                stack.append(ch)
        return "".join(stack)