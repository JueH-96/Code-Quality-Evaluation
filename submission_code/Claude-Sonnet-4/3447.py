class Solution:
    def clearDigits(self, s: str) -> str:
        result = []
        
        for char in s:
            if char.isdigit():
                # Remove the closest non-digit character to the left
                if result:
                    result.pop()
            else:
                # Add non-digit character
                result.append(char)
        
        return ''.join(result)