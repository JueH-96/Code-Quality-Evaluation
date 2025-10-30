class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Compute prefix sums where +1 for possible level, -1 for impossible
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + (1 if possible[i] == 1 else -1)
        
        total_sum = prefix_sum[n]
        
        # We need 1 <= x <= n-1 so that both players play at least one level
        # Check for the smallest x such that Alice's score > Bob's score
        # i.e., prefix_sum[x] > total_sum - prefix_sum[x]
        # which simplifies to 2*prefix_sum[x] > total_sum
        for x in range(1, n):
            if prefix_sum[x] * 2 > total_sum:
                return x
        
        return -1