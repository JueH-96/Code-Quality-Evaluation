class Solution:
    def finalString(self, s: str) -> str:
        result = ""
        
        for char in s:
            if char == 'i':
                result = result[::-1]  # Reverse the string
            else:
                result += char  # Append the character
        
        return result