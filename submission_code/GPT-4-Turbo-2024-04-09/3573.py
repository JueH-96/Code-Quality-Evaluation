class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in word2
        word2_count = Counter(word2)
        n = len(word1)
        m = len(word2)
        
        # If word2 is longer than word1, no valid substrings can exist
        if m > n:
            return 0
        
        # Initialize the count of valid substrings
        valid_count = 0
        
        # Use a sliding window of size m to m+n (up to the length of word1)
        for start in range(n - m + 1):
            for end in range(start + m, n + 1):
                # Extract the substring from word1
                substring = word1[start:end]
                # Count the frequency of each character in the substring
                substring_count = Counter(substring)
                
                # Check if substring can be rearranged to have word2 as a prefix
                valid = True
                for char in word2_count:
                    if substring_count[char] < word2_count[char]:
                        valid = False
                        break
                
                # If valid, increment the count of valid substrings
                if valid:
                    valid_count += 1
        
        return valid_count