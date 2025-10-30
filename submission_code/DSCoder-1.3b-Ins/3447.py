class Solution:
    def clearDigits(self, s: str) -> str:
        result = []
        for i in range(len(s)):
            if not s[i].isdigit():
                result.append(s[i])
            else:
                while i > 0 and s[i-1].isdigit():
                    i -= 1
                else:
                    result.append(s[i])
        return ''.join(result)