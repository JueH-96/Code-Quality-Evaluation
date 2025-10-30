from functools import lru_cache

class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s: str, target: int) -> bool:
            @lru_cache(maxsize=None)
            def helper(pos, current_sum):
                if pos == len(s):
                    return current_sum == target
                for i in range(pos + 1, len(s) + 1):
                    num = int(s[pos:i])
                    if current_sum + num > target:
                        break
                    if helper(i, current_sum + num):
                        return True
                return False
            return helper(0, 0)
        
        total = 0
        for i in range(1, n + 1):
            s = str(i * i)
            if can_partition(s, i):
                total += i * i
        return total