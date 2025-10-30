class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s, target, index, current_sum):
            if index == len(s):
                return current_sum == target
            num = 0
            for i in range(index, len(s)):
                num = num * 10 + int(s[i])
                if current_sum + num > target:
                    break
                if can_partition(s, target, i+1, current_sum + num):
                    return True
            return False
        
        total = 0
        for i in range(1, n+1):
            square = i * i
            if can_partition(str(square), i, 0, 0):
                total += square
        return total