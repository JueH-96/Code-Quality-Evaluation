class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        used = set()
        total_sum = 0
        current = 1
        
        while len(used) < n:
            if k - current not in used:
                used.add(current)
                total_sum += current
            current += 1
        
        return total_sum