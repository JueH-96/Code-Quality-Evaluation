class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        count = 0
        
        # Create a dictionary to store the frequency of characters in word2
        freq = {}
        for c in word2:
            freq[c] = freq.get(c, 0) + 1
        
        # Iterate through the substrings of word1
        for i in range(n):
            for j in range(i, n):
                # Check if the current substring is valid
                substring = word1[i:j+1]
                is_valid = True
                for c in substring:
                    if c not in freq or freq[c] == 0:
                        is_valid = False
                        break
                    freq[c] -= 1
                
                # If the substring is valid, increment the count
                if is_valid:
                    count += 1
                
                # Reset the frequency dictionary
                for c in substring:
                    freq[c] += 1
        
        return count