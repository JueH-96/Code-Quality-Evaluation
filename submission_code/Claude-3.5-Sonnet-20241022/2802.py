class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s: str, target: int, curr_sum: int = 0, start: int = 0) -> bool:
            if start == len(s):
                return curr_sum == target
            
            for i in range(start, len(s)):
                # Skip leading zeros unless it's a single zero
                if s[start] == '0' and i != start:
                    break
                    
                num = int(s[start:i + 1])
                if curr_sum + num > target:
                    break
                    
                if can_partition(s, target, curr_sum + num, i + 1):
                    return True
                    
            return False

        result = 0
        for i in range(1, n + 1):
            square = i * i
            if can_partition(str(square), i):
                result += square
                
        return result