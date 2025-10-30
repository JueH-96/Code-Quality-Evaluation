from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        total_sum = sum(nums)

        # Check if it's possible to make nums a zero array with all queries
        query_coverage = [0] * n
        for l, r in queries:
            query_coverage[l] += 1
            if r + 1 < n:
                query_coverage[r + 1] -= 1

        # Calculate the prefix sum of query_coverage
        for i in range(1, n):
            query_coverage[i] += query_coverage[i - 1]

        # Check if the total decrement possible is enough to make nums a zero array
        total_decrement = sum(min(num, query_coverage[i]) for i, num in enumerate(nums))
        if total_decrement < total_sum:
            return -1

        # Greedily remove queries from the end
        removed_queries = 0
        current_coverage = query_coverage.copy()
        for l, r in reversed(queries):
            # Check if removing this query still allows nums to be a zero array
            temp_coverage = current_coverage[:]
            for i in range(l, r + 1):
                temp_coverage[i] -= 1

            total_decrement = sum(min(num, temp_coverage[i]) for i, num in enumerate(nums))
            if total_decrement >= total_sum:
                removed_queries += 1
                current_coverage = temp_coverage
            else:
                break

        return removed_queries