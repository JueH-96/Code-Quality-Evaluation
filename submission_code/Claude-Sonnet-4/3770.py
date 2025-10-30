class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n, m = len(str1), len(str2)
        result = [''] * (n + m - 1)
        
        def backtrack(pos):
            if pos == n + m - 1:
                return True
            
            # Try characters 'a' to 'z' for lexicographically smallest
            for c in 'abcdefghijklmnopqrstuvwxyz':
                result[pos] = c
                
                # Check if this character violates any constraints
                valid = True
                
                # Check all substrings of length m that include position pos
                for i in range(max(0, pos - m + 1), min(pos + 1, n)):
                    # Substring starts at position i
                    if pos - i == m - 1:  # We just completed a substring of length m
                        substring = ''.join(result[i:i+m])
                        if str1[i] == 'T' and substring != str2:
                            valid = False
                            break
                        elif str1[i] == 'F' and substring == str2:
                            valid = False
                            break
                
                if valid and backtrack(pos + 1):
                    return True
            
            return False
        
        if backtrack(0):
            return ''.join(result)
        else:
            return ""