class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import Counter
        
        # Count frequency of each letter
        freq = Counter(s)
        
        # Find maximum frequency
        max_freq = max(freq.values())
        
        # Keep track of how many times we've seen each character
        seen_count = Counter()
        
        # Build result by including only the last occurrence of max-freq chars
        result = []
        for char in s:
            seen_count[char] += 1
            # Include this character if it's its last occurrence and has max frequency
            if seen_count[char] == freq[char] and freq[char] == max_freq:
                result.append(char)
        
        return ''.join(result)