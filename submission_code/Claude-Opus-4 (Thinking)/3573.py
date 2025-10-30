class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter, defaultdict
        
        target = Counter(word2)
        window = defaultdict(int)
        
        left = 0
        result = 0
        
        for right in range(len(word1)):
            # Add character at right
            window[word1[right]] += 1
            
            # Find the leftmost position where window becomes invalid
            while left <= right:
                # Check if current window is valid
                valid = all(window[c] >= target[c] for c in target)
                
                if not valid:
                    break
                
                # Window is valid, try removing left character
                window[word1[left]] -= 1
                left += 1
            
            # Now window[left-1, right] was the last valid window (if left > 0)
            if left > 0:
                # Restore to make it valid
                left -= 1
                window[word1[left]] += 1
                
                # Count valid substrings ending at right
                if right - left + 1 >= len(word2):
                    # All substrings ending at right and starting from index 0 to left are valid
                    # But they must have length >= len(word2)
                    # So starting index k must satisfy: right - k + 1 >= len(word2)
                    # Which gives: k <= right - len(word2) + 1
                    result += min(left + 1, right - len(word2) + 2)
        
        return result