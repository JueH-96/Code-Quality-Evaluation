class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Frequency array and positions of each character
        freq = [0]*26
        positions = [[] for _ in range(26)]
        
        # Fill frequency and positions
        for i, ch in enumerate(s):
            idx = ord(ch) - ord('a')
            freq[idx] += 1
            positions[idx].append(i)
        
        # Maximum frequency
        max_freq = max(freq)
        
        # Collect (position_of_maxth_occurrence, character) for letters with freq == max_freq
        collect = []
        for c in range(26):
            if freq[c] == max_freq:
                # The index of the max_freq-th occurrence of this character
                pos = positions[c][max_freq - 1]
                collect.append((pos, chr(c + ord('a'))))
        
        # Sort by position and build the resulting string
        collect.sort()
        return ''.join(ch for _, ch in collect)