class Solution:
    def finalString(self, s: str) -> str:
        result = ""
        for ch in s:
            if ch == 'i':
                result = result[::-1]
            else:
                result += ch
        return result