class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        stack = []
        for c in word:
            if stack and stack[-1] == c:
                stack.pop()
            else:
                stack.append(c)
        return len(stack)