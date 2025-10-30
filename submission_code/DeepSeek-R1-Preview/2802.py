class Solution:
    def punishmentNumber(self, n: int) -> int:
        total = 0
        for i in range(1, n + 1):
            square = i * i
            if self.can_split(str(square), i):
                total += square
        return total

    def can_split(self, s: str, target: int) -> bool:
        n = len(s)
        def helper(pos: int, current_sum: int) -> bool:
            if pos == n:
                return current_sum == target
            for i in range(pos + 1, n + 1):
                num = int(s[pos:i])
                if current_sum + num > target:
                    break  # Early termination if sum exceeds target
                if helper(i, current_sum + num):
                    return True
            return False
        return helper(0, 0)