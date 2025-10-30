class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if words[i] in words[j] and words[j][:-len(words[i])] == words[i]:
                    count += 1
                if words[j] in words[i] and words[i][:-len(words[j])] == words[j]:
                    count += 1
        return count