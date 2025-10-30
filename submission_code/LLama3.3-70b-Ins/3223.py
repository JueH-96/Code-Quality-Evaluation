class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_valid(s: str) -> bool:
            """Check if a substring is valid."""
            # Check if each character occurs exactly k times
            char_count = {}
            for char in s:
                if char not in char_count:
                    char_count[char] = 0
                char_count[char] += 1
                if char_count[char] > k:
                    return False
            for count in char_count.values():
                if count != k:
                    return False
            
            # Check if the difference between two adjacent characters is at most 2
            for i in range(len(s) - 1):
                if abs(ord(s[i]) - ord(s[i + 1])) > 2:
                    return False
            
            return True

        count = 0
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                substring = word[i:j]
                if is_valid(substring):
                    count += 1
        
        return count