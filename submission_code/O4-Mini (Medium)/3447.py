class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        for ch in s:
            if ch.isdigit():
                # remove this digit and the closest non-digit to its left
                stack.pop()
            else:
                stack.append(ch)
        return "".join(stack)