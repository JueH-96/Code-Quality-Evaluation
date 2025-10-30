class Solution:
    def maximumLength(self, s: str) -> int:
        max_length = -1
        n = len(s)
        
        # Check each character in the alphabet
        for char in set(s):
            # Gather all segments of the current character
            segments = []
            i = 0
            while i < n:
                if s[i] == char:
                    start = i
                    while i < n and s[i] == char:
                        i += 1
                    segments.append((start, i - 1))
                else:
                    i += 1
            
            # Check each possible length from 1 to n
            for length in range(1, n + 1):
                count = 0
                for start, end in segments:
                    # Number of substrings of this length in the current segment
                    if end - start + 1 >= length:
                        count += (end - start + 2 - length)
                
                # If this length occurs at least thrice, update max_length
                if count >= 3:
                    max_length = max(max_length, length)
        
        return max_length