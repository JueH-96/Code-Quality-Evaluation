class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff = [0] * n
        
        for i in range(n):
            if s1[i] != s2[i]:
                diff[i] = 1
        
        count_ones = sum(diff)
        
        if count_ones % 2 != 0:
            return -1
        
        # Count pairs of adjacent differences
        adjacent_pairs = 0
        for i in range(n - 1):
            if diff[i] == 1 and diff[i + 1] == 1:
                adjacent_pairs += 1
                diff[i] = diff[i + 1] = 0  # Mark them as handled
        
        # Remaining single differences
        single_differences = sum(diff)
        
        # Cost calculation
        cost = adjacent_pairs + single_differences // 2 * x
        
        return cost