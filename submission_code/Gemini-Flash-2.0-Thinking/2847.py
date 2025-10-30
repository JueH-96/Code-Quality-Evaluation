class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        pair_count = 0
        n = len(words)
        for i in range(n):
            for j in range(i + 1, n):
                if words[i] == words[j][::-1]:
                    pair_count += 1
        return pair_count