class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # We'll replace each '?' with the lowest-frequency letter so far,
        # picking the lexicographically smallest letter if there's a tie.
        # This ensures we keep each character's frequency as balanced as possible,
        # which in turn minimizes the total "value" (sum of previous occurrences).
        
        # Convert s to a list for easy in-place modification
        arr = list(s)
        # Frequency array for 'a' to 'z'
        freq = [0]*26
        
        # First, record frequencies of already-determined letters
        for i, ch in enumerate(arr):
            if ch != '?':
                freq[ord(ch) - ord('a')] += 1
                
        # Now fill in the '?' positions
        for i, ch in enumerate(arr):
            if ch == '?':
                # Find the letter with the smallest frequency
                # Ties broken by lexicographical order
                min_count = min(freq)
                for letter in range(26):
                    if freq[letter] == min_count:
                        arr[i] = chr(letter + ord('a'))
                        freq[letter] += 1
                        break
        
        return ''.join(arr)