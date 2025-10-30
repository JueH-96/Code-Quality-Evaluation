class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        n = len(words)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = words[i]
                t = words[j]
                if len(s) <= len(t) and t.startswith(s) and t.endswith(s):
                    count += 1
        return count