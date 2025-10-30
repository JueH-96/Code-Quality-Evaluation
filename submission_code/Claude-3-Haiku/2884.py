class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Create a set of forbidden substrings for faster lookup
        forbidden_set = set(forbidden)
        
        # Initialize variables to keep track of the longest valid substring
        max_length = 0
        current_length = 0
        
        # Iterate through the characters in the word
        for i in range(len(word)):
            # If the current character is the start of a forbidden substring, reset the current length
            if any(word[i:i+j] in forbidden_set for j in range(1, len(word) - i + 1)):
                current_length = 0
            else:
                # Increment the current length
                current_length += 1
                # Update the maximum length if necessary
                max_length = max(max_length, current_length)
        
        return max_length