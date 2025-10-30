class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def countInSegment(s):
            n = len(s)
            count = 0
            
            # Try all possible numbers of distinct characters (1 to 26)
            for num_chars in range(1, 27):
                window_size = num_chars * k
                if window_size > n:
                    break
                
                # Use sliding window for this window size
                char_count = {}
                
                # Initialize first window
                for i in range(window_size):
                    char_count[s[i]] = char_count.get(s[i], 0) + 1
                
                # Check if first window is complete
                if all(count == k for count in char_count.values()):
                    count += 1
                
                # Slide the window
                for i in range(window_size, n):
                    # Add new character
                    char_count[s[i]] = char_count.get(s[i], 0) + 1
                    
                    # Remove old character
                    old_char = s[i - window_size]
                    char_count[old_char] -= 1
                    if char_count[old_char] == 0:
                        del char_count[old_char]
                    
                    # Check if current window is complete
                    if all(count == k for count in char_count.values()):
                        count += 1
            
            return count
        
        # Split word into segments where adjacent chars differ by at most 2
        segments = []
        start = 0
        
        for i in range(1, len(word)):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])
        
        # Count complete substrings in each segment
        total = 0
        for segment in segments:
            if segment:  # Skip empty segments
                total += countInSegment(segment)
        
        return total