class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def check_window(freq, k):
            return all(f == 0 or f == k for f in freq)
            
        def count_in_segment(s, k):
            result = 0
            max_unique = len(s) // k
            
            for unique in range(1, min(27, max_unique + 1)):
                window_size = unique * k
                if window_size > len(s):
                    break
                    
                freq = [0] * 26
                for i in range(window_size):
                    freq[ord(s[i]) - ord('a')] += 1
                    
                if check_window(freq, k):
                    result += 1
                    
                for i in range(window_size, len(s)):
                    freq[ord(s[i - window_size]) - ord('a')] -= 1
                    freq[ord(s[i]) - ord('a')] += 1
                    if check_window(freq, k):
                        result += 1
                        
            return result
            
        segments = []
        start = 0
        
        for i in range(1, len(word)):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])
        
        total = 0
        for segment in segments:
            total += count_in_segment(segment, k)
            
        return total