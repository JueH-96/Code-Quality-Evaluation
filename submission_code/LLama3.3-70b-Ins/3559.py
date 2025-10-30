from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        memo = {}

        def dfs(target: str) -> int:
            if target == "":
                return 0
            if target in memo:
                return memo[target]
            res = float('inf')
            for word in words:
                for i in range(1, len(word) + 1):
                    prefix = word[:i]
                    if target.startswith(prefix):
                        suffix = target[i:]
                        temp = dfs(suffix)
                        if temp != -1:
                            res = min(res, 1 + temp)
            memo[target] = res if res != float('inf') else -1
            return memo[target]

        return dfs(target)