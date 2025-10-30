class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(num):
            num_str = str(num)
            for i in range(1, len(num_str)):
                left = int(num_str[:i])
                right = int(num_str[i:])
                if left + right == int(num ** 0.5):
                    return True
            return False
        
        punishment_sum = 0
        for i in range(1, n + 1):
            square = i * i
            if can_partition(square):
                punishment_sum += square
        return punishment_sum