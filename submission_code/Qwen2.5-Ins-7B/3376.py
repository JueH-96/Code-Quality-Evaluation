class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        n, m = len(wordsContainer), len(wordsQuery)
        ans = [0] * m
        for i in range(m):
            max_len, min_len, idx = 0, float('inf'), -1
            for j in range(n):
                suffix = wordsContainer[j][max(0, len(wordsContainer[j]) - len(wordsQuery[i])):]
                if suffix == wordsQuery[i][-len(suffix):]:
                    if len(wordsContainer[j]) > max_len or (len(wordsContainer[j]) == max_len and len(wordsContainer[j]) < min_len):
                        max_len = len(wordsContainer[j])
                        min_len = len(wordsContainer[j])
                        idx = j
                    elif len(wordsContainer[j]) == max_len and len(wordsContainer[j]) == min_len:
                        idx = min(idx, j)
            ans[i] = idx
        return ans