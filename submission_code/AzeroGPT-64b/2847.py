class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        
        wmap = {}
        res = 0
        for w in words:
            rev = w[::-1]
            if rev in wmap:
                res += 1
                del wmap[rev]
            else:
                wmap[w] = 1
        return res