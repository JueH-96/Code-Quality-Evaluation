class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(s, k):
            if not s:
                return False
            char_counts = {}
            for char in s:
                char_counts[char] = char_counts.get(char, 0) + 1
            for char in char_counts:
                if char_counts[char] != k:
                    return False
            if len(char_counts) == 0:
                return False
            for i in range(len(s) - 1):
                if abs(ord(s[i]) - ord(s[i+1])) > 2:
                    return False
            return True

        count = 0
        n = len(word)
        for i in range(n):
            for j in range(i, n):
                substring = word[i:j+1]
                if is_complete(substring, k):
                    count += 1
        return count