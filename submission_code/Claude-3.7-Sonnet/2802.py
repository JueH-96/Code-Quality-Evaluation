class Solution:
    def punishmentNumber(self, n: int) -> int:
        result = 0
        
        for i in range(1, n + 1):
            square = i * i
            # Check if the square can be partitioned to sum up to i
            if self.can_partition(str(square), 0, i, {}):
                result += square
                
        return result
    
    def can_partition(self, s: str, idx: int, target: int, memo: dict) -> bool:
        # Check if we've computed this subproblem before
        if (idx, target) in memo:
            return memo[(idx, target)]
        
        # Base case: reached end of string
        if idx == len(s):
            return target == 0
        
        # Try all possible partitions starting from idx
        for j in range(idx + 1, len(s) + 1):
            current_val = int(s[idx:j])
            
            # Optimization: if current value exceeds target, no need to check further
            if current_val > target:
                break
                
            # Recursively check if we can partition the remaining string
            if self.can_partition(s, j, target - current_val, memo):
                memo[(idx, target)] = True
                return True
        
        # No valid partition found
        memo[(idx, target)] = False
        return False