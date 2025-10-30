class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(sub):
            if len(sub) % k != 0:
                return False
            char_count = {}
            for char in sub:
                if char in char_count:
                    char_count[char] += 1
                else:
                    char_count[char] = 1
            for count in char_count.values():
                if count != k:
                    return False
            for i in range(1, len(sub)):
                if abs(ord(sub[i]) - ord(sub[i-1])) > 2:
                    return False
            return True

        n = len(word)
        count = 0
        for i in range(n):
            for j in range(i + 1, n + 1):
                if is_complete(word[i:j]):
                    count += 1
        return count