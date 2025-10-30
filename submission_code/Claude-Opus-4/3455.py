class Solution:
    def minimumLength(self, s: str) -> int:
        # Count occurrences of each character
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Calculate minimum length
        min_length = 0
        for count in char_count.values():
            if count >= 3:
                # If count is odd, we can reduce to 1
                # If count is even, we can reduce to 2
                min_length += 1 if count % 2 == 1 else 2
            else:
                # If count is 1 or 2, we cannot reduce it
                min_length += count
        
        return min_length