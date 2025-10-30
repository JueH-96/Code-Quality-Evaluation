class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        prefixes = [0] * n
        suffixes = [0] * n
        for i in range(n):
            prefixes[i] = words[i] + words[i][:-1]
            suffixes[i] = words[i][1:] + words[i]
        prefixes.sort()
        suffixes.sort()
        ans = float('inf')
        for i in range(n):
            ans = min(ans, max(len(prefixes[i]), len(suffixes[i])) + min(len(prefixes[i]), len(suffixes[i])))
        return ans