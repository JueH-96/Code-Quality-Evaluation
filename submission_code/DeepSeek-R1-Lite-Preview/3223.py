class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        # Function to split the string into maximal substrings satisfying adjacency condition
        def get_segments(s):
            segments = []
            start = 0
            for i in range(1, len(s)):
                prev_char = s[i-1]
                curr_char = s[i]
                if abs(ord(curr_char) - ord(prev_char)) > 2:
                    if start < i:
                        segments.append(s[start:i])
                    start = i
            if start < len(s):
                segments.append(s[start:])
            return segments
        
        # Function to count complete substrings in a segment
        def count_complete(segment, k):
            count = 0
            freq = {}
            unique = 0
            exactly_k = 0
            left = 0
            for right in range(len(segment)):
                char = segment[right]
                freq[char] = freq.get(char, 0) + 1
                if freq[char] == 1:
                    unique += 1
                if freq[char] == k:
                    exactly_k += 1
                elif freq[char] == k + 1:
                    exactly_k -= 1
                
                # Shrink window from the left if window size > unique * k
                while right - left + 1 > unique * k:
                    left_char = segment[left]
                    freq[left_char] -= 1
                    if freq[left_char] == k - 1:
                        exactly_k -= 1
                    elif freq[left_char] == k:
                        exactly_k += 1
                    if freq[left_char] == 0:
                        unique -= 1
                        del freq[left_char]
                    left += 1
                
                # Check if current window is valid
                if right - left + 1 == unique * k and exactly_k == unique:
                    count += 1
            return count
        
        # Get all maximal segments
        segments = get_segments(word)
        total = 0
        for segment in segments:
            total += count_complete(segment, k)
        return total