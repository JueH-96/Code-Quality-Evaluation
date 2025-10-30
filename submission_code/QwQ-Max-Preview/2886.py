class Solution:
    def finalString(self, s: str) -> str:
        current = ""
        for char in s:
            if char == 'i':
                current = current[::-1]
            else:
                current += char
        return current