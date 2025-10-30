class Solution:
    def finalString(self, s: str) -> str:
        text = ""
        for ch in s:
            if ch == 'i':
                text = text[::-1]
            else:
                text += ch
        return text