class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Convert s to a list so we can modify characters in place
        arr = list(s)
        
        # Keep frequency of each letter 'a'..'z' in the portion processed so far
        freq = [0]*26
        
        # First pass: update frequencies for existing letters
        for i, ch in enumerate(arr):
            if ch != '?':
                freq[ord(ch) - ord('a')] += 1
        
        # Second pass: replace '?' with the letter that has the smallest frequency so far
        for i, ch in enumerate(arr):
            if ch == '?':
                # Find the letter with the smallest frequency; break ties lexicographically
                min_freq = min(freq)
                for letter_idx in range(26):
                    if freq[letter_idx] == min_freq:
                        arr[i] = chr(ord('a') + letter_idx)
                        freq[letter_idx] += 1
                        break  # stop at the first (lexicographically smallest) letter with min frequency
        
        return "".join(arr)