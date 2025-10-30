class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import Counter
        
        # Count the occurrences of each letter in the string
        count = Counter(s)
        
        # Initialize a list to store the characters to replace '?'
        replacements = []
        
        # Find the minimum cost replacements for '?'
        for _ in range(s.count('?')):
            min_char = min(count, key=lambda x: (count[x], x))
            replacements.append(min_char)
            count[min_char] += 1
        
        # Sort the replacements to ensure lexicographical order
        replacements.sort()
        
        # Create a list from the string to allow modifications
        s_list = list(s)
        replacement_index = 0
        
        # Replace '?' with the sorted replacements
        for i in range(len(s_list)):
            if s_list[i] == '?':
                s_list[i] = replacements[replacement_index]
                replacement_index += 1
        
        return ''.join(s_list)