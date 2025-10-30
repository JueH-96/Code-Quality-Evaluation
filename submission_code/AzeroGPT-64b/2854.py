class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        
        @lru_cache(None)
        def store(s, i, p):
            if i == n:
                return len(s)
            return min(
                store(join(s, words[i]), i + 1, 0), 
                store(join(words[i], s), i + 1, 1)
            )
        
        def join(s1, s2):
            if s1 and s2 and s1[-1] == s2[0]:
                return s1[:-1] + s2[1:]
            return s1 + s2
        
        return store(words[0], 1, 0)