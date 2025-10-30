class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        
        # Check all possible starting points for substrings
        for start in range(n):
            freq = {}
            valid = True
            
            # Expand the substring from the starting point
            for end in range(start, n):
                char = word[end]
                
                # Update frequency of the current character
                if char in freq:
                    freq[char] += 1
                else:
                    freq[char] = 1
                
                # Check if the current substring is valid
                if freq[char] > k:
                    valid = False
                    break
                
                # Check adjacent character condition
                if end > start:
                    prev_char = word[end - 1]
                    if abs(ord(char) - ord(prev_char)) > 2:
                        valid = False
                        break
                
                # If valid and all characters have exactly k occurrences
                if valid and all(v == k for v in freq.values()):
                    count += 1
        
        return count