class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_valid_substring(s):
            # Check if each character in the substring occurs exactly k times
            from collections import Counter
            count = Counter(s)
            for value in count.values():
                if value != k:
                    return False
            
            # Check if the difference between adjacent characters is at most 2
            for i in range(len(s) - 1):
                if abs(ord(s[i]) - ord(s[i + 1])) > 2:
                    return False
            
            return True
        
        n = len(word)
        complete_substrings_count = 0
        
        # Iterate over all possible substrings
        for start in range(n):
            for end in range(start + 1, n + 1):
                substring = word[start:end]
                if is_valid_substring(substring):
                    complete_substrings_count += 1
        
        return complete_substrings_count