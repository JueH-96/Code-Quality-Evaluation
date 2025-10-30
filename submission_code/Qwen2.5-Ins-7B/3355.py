class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix_sum = [0] * n
        suffix_sum = [0] * n
        
        # Calculate prefix sum
        prefix_sum[0] = 1 if possible[0] == 1 else -1
        for i in range(1, n):
            prefix_sum[i] = prefix_sum[i-1] + (1 if possible[i] == 1 else -1)
        
        # Calculate suffix sum
        suffix_sum[-1] = 1 if possible[-1] == 1 else -1
        for i in range(n-2, -1, -1):
            suffix_sum[i] = suffix_sum[i+1] + (1 if possible[i] == 1 else -1)
        
        # Find the minimum number of levels Alice should play
        for i in range(n-1):
            if prefix_sum[i] > suffix_sum[i+1]:
                return i + 1
        
        return -1