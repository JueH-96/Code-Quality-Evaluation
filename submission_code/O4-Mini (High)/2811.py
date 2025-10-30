class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # We can pick at most one from each pair (i, k-i) for i < k.
        # The optimal picks < k are 1..floor(k/2). Any further picks must be >= k.
        half = k // 2
        # If we only need n <= half numbers, we take 1..n
        if n <= half:
            return n * (n + 1) // 2
        # Otherwise, take all 1..half, then the next (n-half) numbers starting at k
        sum_low = half * (half + 1) // 2
        extra = n - half
        # Sum of extra numbers k, k+1, ..., k+extra-1
        sum_high = extra * k + extra * (extra - 1) // 2
        return sum_low + sum_high