class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return "" # Cannot create a good caption
        
        # Create a list to store the count of each character
        count = [0] * 26
        
        # Count the occurrences of each character
        for c in caption:
            count[ord(c) - ord('a')] += 1
        
        # Find the lexicographically smallest good caption
        result = []
        for i in range(n):
            if i > 0 and caption[i] == caption[i-1]:
                continue # Skip if the current character is the same as the previous one
            
            # Try changing the current character to the previous or next character in the alphabet
            if caption[i] != 'a':
                new_char = chr(ord(caption[i]) - 1)
                if count[ord(new_char) - ord('a')] >= 3:
                    result.append(new_char)
                    count[ord(caption[i]) - ord('a')] -= 1
                    count[ord(new_char) - ord('a')] += 1
                    continue
            if caption[i] != 'z':
                new_char = chr(ord(caption[i]) + 1)
                if count[ord(new_char) - ord('a')] >= 3:
                    result.append(new_char)
                    count[ord(caption[i]) - ord('a')] -= 1
                    count[ord(new_char) - ord('a')] += 1
                    continue
            
            # If no valid change is possible, return an empty string
            return ""
        
        return ''.join(result)