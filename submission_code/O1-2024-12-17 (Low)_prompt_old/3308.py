class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import Counter
        
        freq = Counter(s)
        # The number of times we repeat the "remove first occurrence" operation
        # is the maximum frequency of any character.
        max_freq = max(freq.values())
        
        # We will record the index of the max_freq-th occurrence of each character,
        # for those characters whose frequency >= max_freq.
        # That set of positions in ascending order forms the string before the last removal.
        counts = [0]*26  # to track occurrences of each character
        result_indices = []
        
        for i, ch in enumerate(s):
            idx = ord(ch) - ord('a')
            counts[idx] += 1
            # If this is the K-th occurrence of ch (where K = max_freq),
            # it will appear in the final iteration's removal.
            if counts[idx] == max_freq:
                result_indices.append(i)
        
        # Construct the result string from those positions
        # (they are already in ascending order by how we traversed s).
        return ''.join(s[i] for i in result_indices)