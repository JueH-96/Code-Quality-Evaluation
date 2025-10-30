from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        m = defaultdict(list)
        for i, s in enumerate(wordsContainer):
            m[s[::-1]].append((len(s), i))

        res = []
        for q in wordsQuery:
            key = q[::-1]
            t = m.get(key, m[''])
            if not t:
                t = m['']
            res.append(min(t)[1])
        return res