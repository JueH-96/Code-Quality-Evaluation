class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import Counter
        
        # Count the frequency of each character in the string
        char_count = Counter(s)
        
        # List to store the characters that need to be replaced
        replace_chars = []
        
        # Collect all '?' characters
        for char in s:
            if char == '?':
                replace_chars.append(char)
        
        # Sort the characters to replace lexicographically
        replace_chars.sort()
        
        # Replace '?' with the smallest lexicographical characters
        result = list(s)
        replace_index = 0
        for i in range(len(result)):
            if result[i] == '?':
                result[i] = replace_chars[replace_index]
                replace_index += 1
        
        return ''.join(result)