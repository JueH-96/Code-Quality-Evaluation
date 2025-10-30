class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s, target):
            if target == 0:
                return True
            if target < 0:
                return False
            for i in range(1, len(s) + 1):
                if can_partition(s[i:], target - int(s[:i])):
                    return True
            return False

        punishment_number = 0
        for i in range(1, n + 1):
            square = i * i
            if can_partition(str(square), i):
                punishment_number += square
        return punishment_number