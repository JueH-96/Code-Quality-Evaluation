from typing import List
from collections import defaultdict
from functools import lru_cache

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Create a dictionary to store prefixes
        prefixes = defaultdict(set)
        for word in words:
            for i in range(1, len(word) + 1):
                prefixes[word[:i]].add(word)

        # Use lru_cache to memoize the results of the recursive function
        @lru_cache(None)
        def dp(target):
            if not target:
                return 0
            min_count = float('inf')
            for i in range(1, len(target) + 1):
                prefix = target[:i]
                if prefix in prefixes:
                    min_count = min(min_count, 1 + dp(target[i:]))
            return min_count

        result = dp(target)
        return result if result != float('inf') else -1