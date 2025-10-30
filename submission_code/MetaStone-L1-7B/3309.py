class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                s = words[i]
                t = words[j]
                if len(s) > len(t):
                    continue
                if t.startswith(s) and t.endswith(s):
                    count += 1
        return count