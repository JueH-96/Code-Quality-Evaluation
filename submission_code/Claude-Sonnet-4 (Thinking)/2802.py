class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s, target):
            def backtrack(index, current_sum):
                if index == len(s):
                    return current_sum == target
                
                for end in range(index + 1, len(s) + 1):
                    substring = s[index:end]
                    value = int(substring)
                    if backtrack(end, current_sum + value):
                        return True
                return False
            
            return backtrack(0, 0)
        
        result = 0
        for i in range(1, n + 1):
            square = i * i
            if can_partition(str(square), i):
                result += square
        
        return result