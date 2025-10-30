class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        result = 0
        
        # Iterate over all possible starting points of substrings
        for start in range(n):
            freq = {}
            max_freq = 0
            
            # Iterate over all possible ending points of substrings starting from 'start'
            for end in range(start, n):
                char = s[end]
                if char in freq:
                    freq[char] += 1
                else:
                    freq[char] = 1
                
                # Update the maximum frequency of any character in the current substring
                max_freq = max(max_freq, freq[char])
                
                # If the maximum frequency of any character is at least k, count this substring
                if max_freq >= k:
                    result += 1
        
        return result