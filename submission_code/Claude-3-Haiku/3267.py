class Solution:
    def maximumLength(self, s: str) -> int:
        # Create a dictionary to store the count of each character
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Find the longest special substring that occurs at least thrice
        max_length = -1
        for char, count in char_count.items():
            if count >= 3:
                max_length = max(max_length, len(char))
        
        return max_length