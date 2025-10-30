class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s, target, index, current_sum):
            if index == len(s):
                return current_sum == target
            for i in range(index + 1, len(s) + 1):
                num = int(s[index:i])
                if current_sum + num > target:
                    break
                if can_partition(s, target, i, current_sum + num):
                    return True
            return False
        
        total = 0
        for i in range(1, n + 1):
            square = i * i
            if can_partition(str(square), i, 0, 0):
                total += square
        return total