class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        suffixes = defaultdict(list)
        for i, word in enumerate(wordsContainer):
            for j in range(len(word)):
                suffixes[word[j:]].append((i, len(word) - j))
        
        for k, v in suffixes.items():
            suffixes[k] = sorted(v, key=lambda x: (x[1], x[0]))
        
        ans = []
        for word in wordsQuery:
            if word in suffixes:
                ans.append(suffixes[word][0][0])
            else:
                ans.append(suffixes[""][0][0])
        
        return ans