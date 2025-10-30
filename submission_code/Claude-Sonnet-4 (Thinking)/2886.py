class Solution:
    def finalString(self, s: str) -> str:
        result = ""
        for char in s:
            if char == 'i':
                result = result[::-1]  # reverse the string
            else:
                result += char  # append the character
        return result