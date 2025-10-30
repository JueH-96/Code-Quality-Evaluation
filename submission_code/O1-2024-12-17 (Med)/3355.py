class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        # Transform the input array: +1 for 1's, -1 for 0's
        A = [1 if x == 1 else -1 for x in possible]
        
        n = len(A)
        # Compute prefix sums: prefix_sum[i] = sum of A[0..i-1]
        prefix_sum = [0] * (n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + A[i]
        
        total = prefix_sum[n]  # Sum of all elements

        # We want the smallest p in [1..n-1] such that
        # 2 * prefix_sum[p] > total
        for p in range(1, n):
            if 2 * prefix_sum[p] > total:
                return p
        
        return -1