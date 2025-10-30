class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        from functools import lru_cache
        
        def join(x, y):
            if x[-1] == y[0]:
                return x + y[1:]
            else:
                return x + y
        
        @lru_cache(None)
        def dp(i, last):
            if i == len(words):
                return len(last)
            # Option 1: join current word to the last result
            option1 = dp(i + 1, join(last, words[i]))
            # Option 2: join last result to the current word
            option2 = dp(i + 1, join(words[i], last))
            return min(option1, option2)
        
        return dp(1, words[0])