class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Create a set for quick lookup of forbidden substrings
        forbidden_set = set(forbidden)
        
        # Initialize variables to track the longest valid substring
        max_length = 0
        n = len(word)
        
        # Use a sliding window approach to find the longest valid substring
        left = 0
        
        # Iterate over each character in the word
        for right in range(n):
            # Check substrings ending at the current position
            for length in range(1, 11):  # Check up to length 10 (max length of forbidden strings)
                if right - length + 1 < 0:
                    break
                substring = word[right - length + 1:right + 1]
                if substring in forbidden_set:
                    # Move the left pointer to the right of the forbidden substring
                    left = right - length + 2
                    break
            
            # Calculate the current valid substring length
            current_length = right - left + 1
            max_length = max(max_length, current_length)
        
        return max_length