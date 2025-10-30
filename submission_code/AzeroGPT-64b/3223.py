from collections import Counter

class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        count = 0
        N = len(word)

        for start in range(N):
            counter = Counter()
            for end in range(start, N):
                counter[word[end]] += 1

                if all(v == k for v in counter.values()):
                    if all(abs(ord(word[i]) - ord(word[i + 1])) <= 2 for i in range(start, end)):
                        count += 1

        return count