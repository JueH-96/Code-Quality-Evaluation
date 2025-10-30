class Solution:
    def punishmentNumber(self, n: int) -> int:
        total = 0
        
        def can_partition(s, target):
            if not s:
                return target == 0
            for j in range(1, len(s) + 1):
                num = int(s[:j])
                if num > target:
                    break
                if can_partition(s[j:], target - num):
                    return True
            return False
        
        for i in range(1, n + 1):
            square = i * i
            s_str = str(square)
            if can_partition(s_str, i):
                total += square
        return total