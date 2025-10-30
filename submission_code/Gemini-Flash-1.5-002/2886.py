class Solution:
    def finalString(self, s: str) -> str:
        res = ""
        for char in s:
            res += char
            if char == 'i':
                res = res[::-1]
        return res