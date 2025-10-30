class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        def lower_bound(data, val):
            l, r = 0, len(data)
            while l < r:
                mid = l + (r - l) // 2
                if data[mid] < val:
                    l = mid + 1
                else:
                    r = mid
            return l
        
        indices, d = [-1]*len(word2), defaultdict(list)
        for i, c in enumerate(word1):
            d[c].append(i)
        
        for i, c in enumerate(word2):
            if c not in d:
                return []
            val = indices[i-1] + 1 if i > 0 else -1
            pos = lower_bound(d[c], val)
            if pos == len(d[c]):
                if i == len(word2) - 1 and c in d:
                    indices[i] = d[c][-1]
                else:
                    return []
            else:
                indices[i] = d[c][pos]
        
        if len(set(indices)) == 1 and indices[0] == -1:
            return []
        
        return indices