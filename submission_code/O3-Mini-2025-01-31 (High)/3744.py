from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # For any integer n, define f(n) to be the number of times you must replace n with floor(n/4)
        # until it becomes 0. In fact, note that f(n) = k if and only if
        #    4^(k-1) <= n < 4^k.
        # That means:
        #   • If 1 <= n <= 3, then f(n) = 1.
        #   • If 4 <= n <= 15, then f(n) = 2.
        #   • If 16 <= n <= 63, then f(n) = 3, and so on.
        #
        # In one “operation” you can update two numbers (each operation counts as one even though it
        # ‘applies’ one update to two elements). Hence, if for a given array you need to perform a total of U updates (i.e.
        # sum_{n in array} f(n) = U), then the minimal number of operations is ceil(U/2). 
        #
        # For a single query [l, r], the array is all integers from l to r.
        # So total update count is: sum_{n=l}^{r} f(n) = S(r) - S(l-1),
        # where S(x) = sum_{n=1}^{x} f(n).
        #
        # f(n) is constant on ranges where 4^(k-1) <= n < 4^k.
        # We precompute these segments up to the maximum possible r, which is 10^9.
        
        # Precompute segments as tuples of (k, low, high) where for all n in [low, high], f(n)=k.
        segments = []
        k = 1
        low = 1
        max_r = 10**9  # given: r <= 1e9
        while low <= max_r:
            high = low * 4 - 1
            segments.append((k, low, high))
            k += 1
            low *= 4
        
        # Helper function to compute the cumulative sum S(n) = sum_{i=1}^{n} f(i)
        def S(n: int) -> int:
            if n <= 0:
                return 0
            total = 0
            # For each segment, if the segment starts above n, we can stop.
            # Otherwise, count the numbers in the segment that are ≤ n.
            for level, seg_low, seg_high in segments:
                if n < seg_low:
                    break
                # For this segment, only numbers up to n contribute.
                up = seg_high if seg_high <= n else n
                count = up - seg_low + 1
                total += level * count
            return total
        
        ans = 0
        for l, r in queries:
            # Total number of individual updates required for numbers l to r.
            total_updates = S(r) - S(l - 1)
            # Each operation updates 2 numbers, so minimal operations is ceil(total_updates/2)
            ops = (total_updates + 1) // 2
            ans += ops
        return ans