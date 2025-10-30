class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        # Preprocess: For each letter, find lengths of contiguous segments in s
        segments = {chr(c): [] for c in range(ord('a'), ord('z')+1)}
        i = 0
        while i < n:
            char = s[i]
            start = i
            while i < n and s[i] == char:
                i += 1
            segments[char].append(i - start)
        
        # Try all special substring lengths from n down to 1
        # We need at least 3 occurrences for any special substring of the same character.
        for L in range(n, 0, -1):
            for char in segments:
                total_occurrences = 0
                for seg_len in segments[char]:
                    if seg_len >= L:
                        # In a contiguous segment of length seg_len,
                        # the number of substrings of length L is (seg_len - L + 1)
                        total_occurrences += (seg_len - L + 1)
                if total_occurrences >= 3:
                    return L
        return -1