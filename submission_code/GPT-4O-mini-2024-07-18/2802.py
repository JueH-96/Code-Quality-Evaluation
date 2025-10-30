class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s, target):
            if target == 0 and not s:
                return True
            if not s:
                return False
            
            for i in range(1, len(s) + 1):
                part = int(s[:i])
                if part > target:
                    break
                if can_partition(s[i:], target - part):
                    return True
            return False
        
        punishment_sum = 0
        
        for i in range(1, n + 1):
            square = i * i
            if can_partition(str(square), i):
                punishment_sum += square
        
        return punishment_sum