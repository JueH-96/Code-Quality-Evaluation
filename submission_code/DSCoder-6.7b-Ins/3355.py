class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix_sum = [0] * n
        prefix_sum[0] = possible[0]
        for i in range(1, n):
            if possible[i] == 1:
                prefix_sum[i] = prefix_sum[i-1] + 1
            else:
                prefix_sum[i] = prefix_sum[i-1]
        
        min_levels = float('inf')
        for i in range(n-1, -1, -1):
            if i == n-1 or possible[i+1] == 1:
                left_levels = i + 1
            else:
                left_levels = i + 1 - prefix_sum[i]
            right_levels = prefix_sum[i]
            if left_levels > right_levels:
                min_levels = min(min_levels, left_levels)
        
        if min_levels == float('inf'):
            return -1
        else:
            return min_levels