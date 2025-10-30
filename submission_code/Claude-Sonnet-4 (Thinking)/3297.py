class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        
        # Compute all border lengths using KMP's failure function
        def compute_all_border_lengths(s):
            n = len(s)
            if n == 0:
                return {0}
            
            failure = [0] * n
            j = 0
            for i in range(1, n):
                while j > 0 and s[i] != s[j]:
                    j = failure[j - 1]
                if s[i] == s[j]:
                    j += 1
                failure[i] = j
            
            borders = {0}  # Empty border
            j = failure[n - 1]
            while j > 0:
                borders.add(j)
                j = failure[j - 1]
            return borders
        
        borders = compute_all_border_lengths(word)
        
        # Find the minimum t such that the string can revert to its original state
        for t in range(1, n + 1):
            if t * k >= n or (n - t * k) in borders:
                return t
        
        return -1