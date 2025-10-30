class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        stack = []
        for c in word:
            if stack and stack[-1] == c:
                stack.pop()
                continue
            if len(stack) >= 2:
                return 2
            stack.append(c)
        return 0 if len(stack) % 2 == 0 else 1