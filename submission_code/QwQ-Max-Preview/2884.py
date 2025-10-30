class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        L = max(len(s) for s in forbidden) if forbidden else 0
        max_length = 0
        left = 0
        n = len(word)
        
        for right in range(n):
            max_new_left = left  # Track the maximum new left boundary required
            # Check all possible substrings ending at 'right' with length 1 to L
            for l in range(1, L + 1):
                start = right - l + 1
                if start < 0:
                    continue  # Substring starts before the beginning of the word
                if word[start:right+1] in forbidden_set:
                    # Update the new left boundary to exclude this forbidden substring
                    current_new_left = start + 1
                    if current_new_left > max_new_left:
                        max_new_left = current_new_left
            # Update left to the maximum required new left boundary
            left = max_new_left
            # Update the maximum valid substring length
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        
        return max_length