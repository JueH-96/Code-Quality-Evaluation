class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        # Split the word into segments where each consecutive pair has <=2 difference
        segments = []
        n = len(word)
        if n == 0:
            return 0
        start = 0
        for i in range(n - 1):
            curr = word[i]
            next_char = word[i + 1]
            if abs(ord(curr) - ord(next_char)) > 2:
                segments.append(word[start:i+1])
                start = i + 1
        segments.append(word[start:])
        
        total = 0
        
        for seg in segments:
            total += self.process_segment(seg, k)
        
        return total
    
    def process_segment(self, s: str, k: int) -> int:
        count = 0
        n = len(s)
        for m in range(1, 27):
            window_length = m * k
            if window_length > n:
                continue
            # Initialize frequency array and tracking variables
            freq = [0] * 26
            valid = 0  # Number of characters with exactly k occurrences
            distinct = 0  # Number of distinct characters in the current window
            
            # Initialize the first window
            for i in range(window_length):
                c = s[i]
                idx = ord(c) - ord('a')
                freq[idx] += 1
                if freq[idx] == 1:
                    distinct += 1
                if freq[idx] == k:
                    valid += 1
                elif freq[idx] == k + 1:
                    valid -= 1
            
            if valid == m and distinct == m:
                count += 1
            
            # Slide the window through the rest of the segment
            for i in range(1, n - window_length + 1):
                # Remove the leftmost character of the previous window
                left_char = s[i - 1]
                left_idx = ord(left_char) - ord('a')
                freq[left_idx] -= 1
                if freq[left_idx] == k - 1:
                    valid -= 1
                elif freq[left_idx] == k:
                    valid += 1
                if freq[left_idx] == 0:
                    distinct -= 1
                
                # Add the new right character of the current window
                right_char = s[i + window_length - 1]
                right_idx = ord(right_char) - ord('a')
                freq[right_idx] += 1
                if freq[right_idx] == 1:
                    distinct += 1
                if freq[right_idx] == k:
                    valid += 1
                elif freq[right_idx] == k + 1:
                    valid -= 1
                
                # Check if the current window meets the criteria
                if valid == m and distinct == m:
                    count += 1
        
        return count