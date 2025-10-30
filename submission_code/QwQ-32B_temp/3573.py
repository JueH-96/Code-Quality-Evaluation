from collections import defaultdict

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        target = defaultdict(int)
        for c in word2:
            target[c] += 1
        required = len(target)
        m = len(word2)
        n = len(word1)
        if n < m:
            return 0
        
        current_counts = defaultdict(int)
        matches = 0
        result = 0
        left = 0
        
        for right in range(n):
            c = word1[right]
            if c in target:
                current_counts[c] += 1
                if current_counts[c] == target[c]:
                    matches += 1
                elif current_counts[c] == target[c] + 1:
                    # This character's count exceeded the target, so no change to matches
                    pass
            
            # Now check if the window can be shrunk while maintaining validity
            while left <= right and matches == required and (right - left + 1) >= m:
                # Add the contribution of this valid window
                result += (n - right)
                # Move left forward
                left_char = word1[left]
                if left_char in target:
                    current_counts[left_char] -= 1
                    if current_counts[left_char] == target[left_char] - 1:
                        matches -= 1
                left += 1
        
        return result