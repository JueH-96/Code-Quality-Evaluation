class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for i in range(n):
            for j in range(i + 1, n):
                if words[i] == words[j][:len(words[i])] and words[i] == words[j][-len(words[i]):]:
                    count += 1
        return count