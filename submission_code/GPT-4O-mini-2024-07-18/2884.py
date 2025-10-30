class Solution:
    def longestValidSubstring(self, word: str, forbidden: list[str]) -> int:
        forbidden_set = set(forbidden)
        max_length = 0
        n = len(word)
        
        # Use a sliding window approach
        left = 0
        
        for right in range(n):
            # Check all substrings ending at 'right'
            for length in range(1, 11):  # Check substrings of length 1 to 10
                if right - length + 1 >= 0:
                    substring = word[right - length + 1:right + 1]
                    if substring in forbidden_set:
                        left = right - length + 2  # Move left pointer past the forbidden substring
                        break
            
            max_length = max(max_length, right - left + 1)
        
        return max_length