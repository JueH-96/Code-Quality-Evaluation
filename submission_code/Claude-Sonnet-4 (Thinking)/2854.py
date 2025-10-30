class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(i, first, last):
            if i == len(words):
                return 0
            
            word = words[i]
            
            # Option 1: join(current_string, word)
            save1 = 1 if last == word[0] else 0
            cost1 = len(word) - save1 + dfs(i+1, first, word[-1])
            
            # Option 2: join(word, current_string)
            save2 = 1 if word[-1] == first else 0
            cost2 = len(word) - save2 + dfs(i+1, word[0], last)
            
            return min(cost1, cost2)
        
        return len(words[0]) + dfs(1, words[0][0], words[0][-1])