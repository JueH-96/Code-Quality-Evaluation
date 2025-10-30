class Solution:
    def minimizeStringValue(self, s: str) -> str:
        result = []
        count = {}
        
        for char in s:
            if char == '?':
                # Try to replace '?' with the smallest possible letter
                for replacement in 'abcdefghijklmnopqrstuvwxyz':
                    if count.get(replacement, 0) == 0:
                        result.append(replacement)
                        count[replacement] = count.get(replacement, 0) + 1
                        break
            else:
                result.append(char)
                count[char] = count.get(char, 0) + 1
        
        return ''.join(result)