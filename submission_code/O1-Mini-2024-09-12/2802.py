class Solution:
    def punishmentNumber(self, n: int) -> int:
        from functools import lru_cache

        def can_partition(s: str, target: int) -> bool:
            @lru_cache(None)
            def dfs(pos: int, remaining: int) -> bool:
                if pos == len(s):
                    return remaining == 0
                for i in range(pos + 1, len(s) + 1):
                    num = int(s[pos:i])
                    if dfs(i, remaining - num):
                        return True
                return False

            return dfs(0, target)

        total = 0
        for i in range(1, n + 1):
            square = i * i
            if can_partition(str(square), i):
                total += square
        return total