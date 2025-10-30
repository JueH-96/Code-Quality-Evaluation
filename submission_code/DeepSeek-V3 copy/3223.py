class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(s):
            if len(s) % k != 0:
                return False
            char_counts = {}
            for char in s:
                if char in char_counts:
                    char_counts[char] += 1
                else:
                    char_counts[char] = 1
            for count in char_counts.values():
                if count != k:
                    return False
            for i in range(1, len(s)):
                if abs(ord(s[i]) - ord(s[i-1])) > 2:
                    return False
            return True
        
        n = len(word)
        count = 0
        for i in range(n):
            for j in range(i+1, n+1):
                substring = word[i:j]
                if is_complete(substring):
                    count += 1
        return count