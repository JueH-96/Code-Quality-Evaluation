class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        prefix = [0] * n
        suffix = [0] * n
        for i in range(1, n):
            j = prefix[i-1]
            while j > 0 and word[i] != word[j]:
                j = prefix[j-1]
            if word[i] == word[j]:
                j += 1
            prefix[i] = j
        for i in range(n-2, -1, -1):
            j = suffix[i+1]
            while j > 0 and word[i] != word[j]:
                j = suffix[j-1]
            if word[i] == word[j]:
                j += 1
            suffix[i] = j
        max_len = 0
        for i in range(n-k+1):
            if i > 0 and prefix[i-1] > 0 and suffix[i+k] > 0 and word[prefix[i-1]-1] == word[n-suffix[i+k]]:
                max_len = max(max_len, prefix[i-1]+suffix[i+k])
        return (n-max_len)//k