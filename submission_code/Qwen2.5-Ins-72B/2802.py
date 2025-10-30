class Solution:
    def punishmentNumber(self, n: int) -> int:
        
        def can_partition(num_str, target):
            if int(num_str) == target:
                return True
            for i in range(1, len(num_str)):
                left, right = num_str[:i], num_str[i:]
                if int(left) < target and can_partition(right, target - int(left)):
                    return True
            return False
        
        punishment_sum = 0
        for i in range(1, n + 1):
            square_str = str(i * i)
            if can_partition(square_str, i):
                punishment_sum += i * i
        
        return punishment_sum