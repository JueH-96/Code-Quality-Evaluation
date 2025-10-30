class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def count_complete_in_segment(segment, k):
            count = 0
            n = len(segment)
            
            # Try all possible numbers of distinct characters (1 to 26)
            for num_chars in range(1, 27):
                window_size = num_chars * k
                if window_size > n:
                    break
                
                # Use sliding window of size window_size
                freq = {}
                
                # Initialize window
                for i in range(window_size):
                    freq[segment[i]] = freq.get(segment[i], 0) + 1
                
                # Check if current window is valid
                if len(freq) == num_chars and all(f == k for f in freq.values()):
                    count += 1
                
                # Slide the window
                for i in range(window_size, n):
                    # Add new character
                    freq[segment[i]] = freq.get(segment[i], 0) + 1
                    
                    # Remove old character
                    old_char = segment[i - window_size]
                    freq[old_char] -= 1
                    if freq[old_char] == 0:
                        del freq[old_char]
                    
                    # Check if current window is valid
                    if len(freq) == num_chars and all(f == k for f in freq.values()):
                        count += 1
            
            return count
        
        # Split the word into segments
        segments = []
        start = 0
        
        for i in range(1, len(word)):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])
        
        # Count complete substrings in each segment
        total_count = 0
        for segment in segments:
            total_count += count_complete_in_segment(segment, k)
        
        return total_count