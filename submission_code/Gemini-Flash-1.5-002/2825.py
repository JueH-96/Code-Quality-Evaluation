class Solution:
    def minimizedStringLength(self, s: str) -> int:
        stack = []
        seen = set()
        for char in s:
            if char in seen:
                continue
            while stack and stack[-1] < char:
                seen.remove(stack.pop())
            stack.append(char)
            seen.add(char)
        return len(stack)