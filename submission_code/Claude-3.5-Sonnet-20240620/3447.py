class Solution:
    def clearDigits(self, s: str) -> str:
        result = list(s)
        i = 0
        while i < len(result):
            if result[i].isdigit():
                if i > 0:
                    result.pop(i-1)
                    result.pop(i-1)
                    i -= 1
                else:
                    result.pop(i)
            else:
                i += 1
        return ''.join(result)