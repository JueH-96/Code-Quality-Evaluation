class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s, target):
            if target == 0:
                return True
            for i in range(1, len(s) + 1):
                if int(s[:i]) <= target and can_partition(s[i:], target - int(s[:i])):
                    return True
            return False
        
        punishment_sum = 0
        for i in range(1, n + 1):
            if can_partition(str(i * i), i):
                punishment_sum += i * i
        return punishment_sum