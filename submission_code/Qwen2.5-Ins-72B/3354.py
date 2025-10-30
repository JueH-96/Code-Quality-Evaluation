class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import Counter
        
        # Count the occurrences of each character in the string
        count = Counter(s)
        
        # Initialize a list to keep track of the number of occurrences of each letter
        letter_count = [0] * 26
        
        # Update the letter_count array with the current counts
        for char in count:
            if char != '?':
                letter_count[ord(char) - ord('a')] = count[char]
        
        # List to store the characters to be used for replacing '?'
        replacements = []
        
        # Find the characters to replace '?' with
        for _ in range(count['?']):
            # Find the letter with the minimum count
            min_count = min(letter_count)
            min_index = letter_count.index(min_count)
            replacements.append(chr(min_index + ord('a')))
            letter_count[min_index] += 1
        
        # Sort the replacements to ensure lexicographically smallest result
        replacements.sort()
        
        # Replace '?' in the original string
        result = list(s)
        j = 0
        for i in range(len(result)):
            if result[i] == '?':
                result[i] = replacements[j]
                j += 1
        
        return ''.join(result)