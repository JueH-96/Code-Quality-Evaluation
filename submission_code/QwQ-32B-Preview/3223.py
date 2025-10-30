class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        if n < k:
            return 0
        
        # Find positions where adjacency condition is violated
        violation = []
        for i in range(n - 1):
            if abs(ord(word[i]) - ord(word[i + 1])) > 2:
                violation.append(i)
        
        # Split the string into segments based on violation positions
        segments = []
        start = 0
        for pos in violation:
            end = pos + 1
            segments.append(word[start:end])
            start = end
        segments.append(word[start:])
        
        total_count = 0
        for segment in segments:
            seg_len = len(segment)
            unique_chars = set(segment)
            max_m = len(unique_chars)
            
            for m in range(1, max_m + 1):
                window_size = m * k
                if window_size > seg_len:
                    continue
                
                freq = {}
                for i in range(window_size):
                    freq[segment[i]] = freq.get(segment[i], 0) + 1
                
                if len(freq) == m and all(v == k for v in freq.values()):
                    total_count += 1
                
                for i in range(window_size, seg_len):
                    # Remove the element going out of the window
                    outgoing_char = segment[i - window_size]
                    freq[outgoing_char] -= 1
                    if freq[outgoing_char] == 0:
                        del freq[outgoing_char]
                    
                    # Add the new element into the window
                    incoming_char = segment[i]
                    freq[incoming_char] = freq.get(incoming_char, 0) + 1
                    
                    if len(freq) == m and all(v == k for v in freq.values()):
                        total_count += 1
        
        return total_count