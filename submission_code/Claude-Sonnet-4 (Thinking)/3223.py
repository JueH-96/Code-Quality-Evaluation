class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def countInSegment(segment):
            n = len(segment)
            count = 0
            
            # Try all possible numbers of distinct characters
            for num_chars in range(1, 27):  # 1 to 26 distinct characters
                length = num_chars * k
                if length > n:
                    break
                
                # Check all substrings of this length
                for i in range(n - length + 1):
                    char_count = {}
                    for j in range(i, i + length):
                        c = segment[j]
                        char_count[c] = char_count.get(c, 0) + 1
                    
                    # Check if each character appears exactly k times
                    if len(char_count) == num_chars and all(cnt == k for cnt in char_count.values()):
                        count += 1
            
            return count
        
        # Split word into segments where adjacent characters have difference ≤ 2
        segments = []
        start = 0
        
        for i in range(1, len(word)):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])
        
        total_count = 0
        for segment in segments:
            total_count += countInSegment(segment)
        
        return total_count