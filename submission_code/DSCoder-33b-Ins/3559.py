class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        from collections import defaultdict
        from functools import lru_cache

        prefixes = defaultdict(list)
        for word in words:
            for i in range(1, len(word) + 1):
                prefixes[word[:i]].append(word)

        @lru_cache(None)
        def dp(i):
            if i == len(target):
                return 0
            min_count = float('inf')
            for word in prefixes[target[i:]]:
                if word.startswith(target[i:i + len(word)]):
                    min_count = min(min_count, 1 + dp(i + len(word)))
            return min_count

        res = dp(0)
        return res if res != float('inf') else -1