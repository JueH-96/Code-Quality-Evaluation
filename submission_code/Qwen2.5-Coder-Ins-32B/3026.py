class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        used = set()
        current = 1
        total_sum = 0
        
        while len(used) < n:
            if target - current not in used:
                used.add(current)
                total_sum += current
            current += 1
        
        return total_sum % MOD