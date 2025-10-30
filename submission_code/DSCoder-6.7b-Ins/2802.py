class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_valid(i: int, target: int) -> bool:
            if i == 0:
                return target == 0
            if target < 0:
                return False
            if i % 10 > target:
                return False
            return is_valid(i // 10, target - i % 10) or is_valid(i // 10, target)

        res = 0
        for i in range(1, n + 1):
            if i * i > 10 * n:
                break
            if is_valid(i * i, i):
                res += i * i
        return res