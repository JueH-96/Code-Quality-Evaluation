class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n, m = len(str1), len(str2)
        result = []
        
        for i in range(n + m - 1):
            # Try each character from 'a' to 'z'
            found_valid = False
            for char_val in range(ord('a'), ord('z') + 1):
                char = chr(char_val)
                result.append(char)
                
                # Check if all relevant conditions are satisfied
                valid = True
                for j in range(max(0, i - m + 1), min(i + 1, n)):
                    # Only check conditions where we have enough characters
                    if j + m - 1 <= i:
                        substring = ''.join(result[j:j+m])
                        if (str1[j] == 'T' and substring != str2) or (str1[j] == 'F' and substring == str2):
                            valid = False
                            break
                
                if valid:
                    found_valid = True
                    break
                
                result.pop()  # Remove the invalid character
            
            if not found_valid:  # Couldn't find a valid character for this position
                return ""
        
        return ''.join(result)