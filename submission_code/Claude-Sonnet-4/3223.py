class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def countInSegment(s):
            if not s:
                return 0
            
            count = 0
            n = len(s)
            
            # Try all possible numbers of distinct characters (1 to 26)
            for num_chars in range(1, 27):
                window_size = num_chars * k
                if window_size > n:
                    break
                
                # Sliding window of size window_size
                char_count = {}
                
                # Initialize first window
                for i in range(window_size):
                    char_count[s[i]] = char_count.get(s[i], 0) + 1
                
                # Check if first window is valid
                if (len(char_count) == num_chars and 
                    all(count == k for count in char_count.values())):
                    count += 1
                
                # Slide the window
                for i in range(window_size, n):
                    # Remove leftmost character
                    left_char = s[i - window_size]
                    char_count[left_char] -= 1
                    if char_count[left_char] == 0:
                        del char_count[left_char]
                    
                    # Add rightmost character
                    right_char = s[i]
                    char_count[right_char] = char_count.get(right_char, 0) + 1
                    
                    # Check if current window is valid
                    if (len(char_count) == num_chars and 
                        all(count == k for count in char_count.values())):
                        count += 1
            
            return count
        
        # Split word into valid segments
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
            total += countInSegment(segment)
        
        return total