from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        """
        Counts pairs (i, j) with i < j such that words[i] is simultaneously a
        prefix and a suffix of words[j].
        """
        n = len(words)
        ans = 0
        
        # Iterate over all ordered pairs (i, j) with i < j
        for i in range(n):
            first = words[i]
            for j in range(i + 1, n):
                second = words[j]
                
                # Quick length check (strictly optional but saves a bit of work)
                if len(first) > len(second):
                    continue
                
                # Check prefix and suffix conditions
                if second.startswith(first) and second.endswith(first):
                    ans += 1
                    
        return ans