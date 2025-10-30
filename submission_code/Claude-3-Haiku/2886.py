class Solution:
    def finalString(self, s: str) -> str:
        result = ""
        reversed_part = ""
        for char in s:
            if char == 'i':
                result = reversed_part + result
                reversed_part = ""
            else:
                reversed_part += char
        return result + reversed_part