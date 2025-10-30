class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s: str, target: int) -> bool:
            if target < 0:
                return False
            if int(s) == target:
                return True
            for i in range(1, len(s)):
                if can_partition(s[i:], target - int(s[:i])):
                    return True
            return False

        punishment_sum = 0
        for i in range(1, n + 1):
            if can_partition(str(i * i), i):
                punishment_sum += i * i
        return punishment_sum