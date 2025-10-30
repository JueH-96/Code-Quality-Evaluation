class Solution:
    def punishmentNumber(self, n: int) -> int:
        # Helper DFS to check if s (digits of square) can be partitioned
        # into pieces summing to target.
        def dfs(s: str, target: int, idx: int, cur_sum: int) -> bool:
            if idx == len(s):
                return cur_sum == target
            num = 0
            # Extend the next substring from idx to j inclusive
            for j in range(idx, len(s)):
                num = num * 10 + (ord(s[j]) - ord('0'))
                # Prune if sum exceeds target
                if cur_sum + num > target:
                    break
                if dfs(s, target, j + 1, cur_sum + num):
                    return True
            return False

        total = 0
        for i in range(1, n + 1):
            sq_str = str(i * i)
            if dfs(sq_str, i, 0, 0):
                total += i * i
        return total