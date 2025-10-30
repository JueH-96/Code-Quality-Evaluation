class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Count frequency of each character
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        # Find maximum frequency
        max_freq = max(freq.values())
        
        # Find all characters with maximum frequency
        max_chars = set()
        for char, count in freq.items():
            if count == max_freq:
                max_chars.add(char)
        
        # Build result by finding the last occurrence of each max frequency character
        # We need to preserve their relative order from the original string
        result = []
        # Traverse from right to left to find last occurrences
        seen = set()
        for i in range(len(s) - 1, -1, -1):
            if s[i] in max_chars and s[i] not in seen:
                result.append(s[i])
                seen.add(s[i])
        
        # Reverse since we built from right to left
        return ''.join(reversed(result))