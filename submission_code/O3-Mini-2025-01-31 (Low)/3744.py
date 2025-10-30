from math import ceil

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # Precompute intervals for numbers with the same number of individual operations.
        # For any x, let f(x) be the minimum number of times you must replace x with floor(x/4)
        # until it becomes 0. This is exactly the minimal k such that:
        #     floor(x / 4^k) == 0
        # This happens when x < 4^k.
        # In particular, for k ≥ 1, all numbers x in the interval [4^(k-1), 4^k - 1] require exactly k steps.
        #
        # Our plan:
        # For each query [l, r], we have an array nums = [l, l+1, ..., r].
        # The total individual operations required for this array is the sum over x in [l, r] of f(x).
        # However, one allowed operation can process two numbers concurrently,
        # so the minimum number of operations for the query is ceil(total_individual_steps/2).
        #
        # Since the range [l, r] could be large (up to 1e9) we can’t iterate through every number.
        # Instead we aggregate counts over intervals that completely lie inside [l, r].
        
        # Precompute group intervals: Each group for a given k has x in [4^(k-1), 4^k - 1].
        groups = []
        k = 1
        power = 1  # 4^(k-1)
        # Since numbers go up to 1e9, we only need groups while power <= 1e9.
        while power <= 10**9:
            L = power
            R = power * 4 - 1   # upper bound for this operation count
            groups.append((L, R, k))
            power *= 4
            k += 1

        total_operations_sum = 0
        for l, r in queries:
            total_indiv_ops = 0  # total individual operations needed for array [l, r]
            # For each group interval [L, R] with cost k:
            for (L, R, k) in groups:
                # The intersection of [l, r] with [L, R] is:
                if r < L or l > R:
                    continue  # no intersection
                low = max(l, L)
                high = min(r, R)
                count = high - low + 1  # number of elements in the intersection
                total_indiv_ops += count * k
            # In one allowed operation, you can fix 2 individual operations (by processing 2 numbers simultaneously).
            # Hence, the minimal operations for this query is ceil(total_indiv_ops / 2).
            query_ops = (total_indiv_ops + 1) // 2
            total_operations_sum += query_ops
        
        return total_operations_sum