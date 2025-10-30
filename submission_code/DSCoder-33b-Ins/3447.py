class Solution:
    def clearDigits(self, s: str) -> str:
        result = []
        for c in s:
            if not c.isdigit():
                result.append(c)
            elif result:
                result.pop()
        return ''.join(result)