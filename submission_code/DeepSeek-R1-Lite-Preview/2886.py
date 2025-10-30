class Solution:
    def finalString(self, s: str) -> str:
        result = []
        for char in s:
            if char != 'i':
                result.append(char)
            else:
                result.reverse()
        return ''.join(result)