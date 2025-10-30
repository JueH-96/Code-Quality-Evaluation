class Solution:
    def punishmentNumber(self, n: int) -> int:
        def check(i: int, target: int, s: str) -> bool:
            if target < 0:
                return False
            if not s:
                return target == 0
            for j in range(1, len(s) + 1):
                if check(i, target - int(s[:j]), s[j:]):
                    return True
            return False

        ans = 0
        for i in range(1, n + 1):
            square = str(i * i)
            if check(i, i, square):
                ans += i * i
        return ans