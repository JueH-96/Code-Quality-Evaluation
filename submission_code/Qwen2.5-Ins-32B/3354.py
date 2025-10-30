class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import Counter
        
        # Count the frequency of each character in the string
        char_count = Counter(s)
        # Remove the count of '?'
        del char_count['?']
        
        # List to store the characters that will replace '?'
        replacements = []
        
        # Iterate over each '?' in the string
        for _ in range(s.count('?')):
            # Find the character with the minimum count
            min_char = min(char_count, key=char_count.get, default='a')
            # Append the character to the replacements list
            replacements.append(min_char)
            # Increment the count of the chosen character
            char_count[min_char] += 1
        
        # Sort the replacements list to ensure lexicographical order
        replacements.sort()
        
        # Replace '?' in the original string with characters from replacements list
        result = []
        j = 0
        for char in s:
            if char == '?':
                result.append(replacements[j])
                j += 1
            else:
                result.append(char)
        
        return ''.join(result)