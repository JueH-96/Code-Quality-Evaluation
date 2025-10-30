class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(num_str, target):
            if target == 0:
                return True
            if not num_str:
                return False
            for i in range(1, len(num_str) + 1):
                if int(num_str[:i]) <= target and can_partition(num_str[i:], target - int(num_str[:i])):
                    return True
            return False
        
        punishment_sum = 0
        for i in range(1, n + 1):
            square_str = str(i * i)
            if can_partition(square_str, i):
                punishment_sum += i * i
        
        return punishment_sum