class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(sub):
            char_count = {}
            for char in sub:
                if char not in char_count:
                    char_count[char] = 0
                char_count[char] += 1
            for count in char_count.values():
                if count != k:
                    return False
            return True

        def is_valid(sub):
            for i in range(1, len(sub)):
                if abs(ord(sub[i]) - ord(sub[i - 1])) > 2:
                    return False
            return True

        def find_substrings(word, k):
            n = len(word)
            count = 0
            for i in range(n):
                for j in range(i + k, n + 1, k):
                    sub = word[i:j]
                    if len(sub) % k == 0 and is_complete(sub) and is_valid(sub):
                        count += 1
            return count

        return find_substrings(word, k)