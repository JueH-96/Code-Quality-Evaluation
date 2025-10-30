class Solution:
    def minLength(self, s: str) -> int:
        result = []
        for char in s:
            result.append(char)
            while True:
                modified = False
                if len(result) >= 2 and "".join(result[-2:]) == "AB":
                    result.pop()
                    result.pop()
                    modified = True
                elif len(result) >= 2 and "".join(result[-2:]) == "CD":
                    result.pop()
                    result.pop()
                    modified = True
                if not modified:
                    break
        return len(result)