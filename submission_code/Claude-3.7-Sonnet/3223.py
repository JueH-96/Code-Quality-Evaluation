class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        result = 0
        
        # Divide the word into segments
        segments = []
        start = 0
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])
        
        # For each segment, count the number of valid complete substrings
        for segment in segments:
            result += self.countValidSubstrings(segment, k)
        
        return result

    def countValidSubstrings(self, segment: str, k: int) -> int:
        count = 0
        
        # For each possible number of unique characters
        for num_unique in range(1, 27):  # Maximum 26 lowercase letters
            window_size = num_unique * k
            if window_size > len(segment):
                break
            
            # Initialize the counter for the first window
            char_count = {}
            for i in range(min(window_size, len(segment))):
                char_count[segment[i]] = char_count.get(segment[i], 0) + 1
            
            # Check if the first window is valid
            if len(char_count) == num_unique and all(count == k for count in char_count.values()):
                count += 1
            
            # Slide the window
            for i in range(window_size, len(segment)):
                char_count[segment[i]] = char_count.get(segment[i], 0) + 1
                char_count[segment[i-window_size]] -= 1
                if char_count[segment[i-window_size]] == 0:
                    del char_count[segment[i-window_size]]
                
                # Check if the current window is valid
                if len(char_count) == num_unique and all(count == k for count in char_count.values()):
                    count += 1
        
        return count