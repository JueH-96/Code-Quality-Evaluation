class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        if n < 2:
            return -1  # Though constraints ensure n >= 2
        
        prefix = [0]
        current = 0
        for p in possible:
            current += (2 * p - 1)
            prefix.append(current)
        
        total_sum = prefix[-1]
        
        for k in range(1, n):
            if 2 * prefix[k] > total_sum:
                return k
        
        return -1