class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        
        # If k is 0, we can always return true
        if k == 0:
            return True
            
        # Create a frequency map for each character
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
            
        # Find all possible special substrings
        special_substrings = []
        start = 0
        current_chars = set()
        current_freq = {}
        
        for end in range(n):
            char = s[end]
            current_chars.add(char)
            current_freq[char] = current_freq.get(char, 0) + 1
            
            # Check if we need to shrink the window
            while start <= end:
                # Check if current substring is special
                is_special = True
                for c in current_chars:
                    if current_freq[c] != char_count[c]:
                        is_special = False
                        break
                
                if is_special and end - start + 1 < n:  # Make sure it's not the entire string
                    special_substrings.append((start, end))
                
                # Shrink window
                start_char = s[start]
                current_freq[start_char] -= 1
                if current_freq[start_char] == 0:
                    current_chars.remove(start_char)
                start += 1
            
            start = end + 1
            current_chars.clear()
            current_freq.clear()
        
        # Find maximum number of non-overlapping intervals
        if not special_substrings:
            return False
            
        # Sort by end position
        special_substrings.sort(key=lambda x: x[1])
        
        count = 1
        last_end = special_substrings[0][1]
        
        for i in range(1, len(special_substrings)):
            if special_substrings[i][0] > last_end:
                count += 1
                last_end = special_substrings[i][1]
                if count >= k:
                    return True
        
        return count >= k