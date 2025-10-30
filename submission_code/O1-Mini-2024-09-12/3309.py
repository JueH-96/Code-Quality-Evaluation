class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for i in range(n):
            prefix_suffix = words[i]
            prefix_len = len(prefix_suffix)
            for j in range(i + 1, n):
                word = words[j]
                if len(prefix_suffix) > len(word):
                    continue
                if word.startswith(prefix_suffix) and word.endswith(prefix_suffix):
                    count += 1
        return count