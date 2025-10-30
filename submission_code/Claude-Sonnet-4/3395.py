class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        
        # Try all possible lengths that divide n evenly
        for length in range(1, n + 1):
            if n % length != 0:
                continue
            
            # Split s into segments of this length
            segments = []
            for i in range(0, n, length):
                segments.append(s[i:i + length])
            
            # Check if all segments are anagrams of each other
            # by comparing their character frequencies
            first_segment_freq = {}
            for char in segments[0]:
                first_segment_freq[char] = first_segment_freq.get(char, 0) + 1
            
            all_anagrams = True
            for segment in segments[1:]:
                segment_freq = {}
                for char in segment:
                    segment_freq[char] = segment_freq.get(char, 0) + 1
                
                if segment_freq != first_segment_freq:
                    all_anagrams = False
                    break
            
            if all_anagrams:
                return length
        
        return n