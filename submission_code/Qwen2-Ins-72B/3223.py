from collections import Counter

class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(sub):
            counts = Counter(sub)
            for i in range(len(sub) - 1):
                if abs(ord(sub[i]) - ord(sub[i + 1])) > 2:
                    return False
            for count in counts.values():
                if count != k:
                    return False
            return True

        count = 0
        for i in range(len(word)):
            for j in range(i + k, len(word) + 1, k):
                if is_complete(word[i:j]):
                    count += 1
        return count