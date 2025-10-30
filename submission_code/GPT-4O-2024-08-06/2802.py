class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(num_str, target):
            # Helper function to check if num_str can be partitioned to sum to target
            if target == 0:
                return True
            if not num_str:
                return False
            
            for i in range(1, len(num_str) + 1):
                part = int(num_str[:i])
                if part > target:
                    break
                if can_partition(num_str[i:], target - part):
                    return True
            return False
        
        punishment_sum = 0
        
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            if can_partition(square_str, i):
                punishment_sum += square
        
        return punishment_sum