class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)

        def can_zero_out(indices):
            selected_queries = [queries[i] for i in indices]
            coverage = [0] * n
            for l, r in selected_queries:
                for i in range(l, r + 1):
                    coverage[i] += 1
            for i in range(n):
                if coverage[i] < nums[i]:
                    return False
            return True

        # Check if it's possible to zero out nums using all queries
        all_indices = list(range(m))
        if not can_zero_out(all_indices):
            return -1

        # Binary search for the maximum number of removable queries
        low = 0
        high = m
        max_removable = -1

        while low <= high:
            removals = (low + high) // 2
            keep = m - removals

            found_subset = False
            if keep == 0:
                if all(n == 0 for n in nums):
                    found_subset = True
            elif keep > 0:
                from itertools import combinations
                for indices in combinations(range(m), keep):
                    if can_zero_out(list(indices)):
                        found_subset = True
                        break

            if found_subset:
                max_removable = removals
                low = removals + 1
            else:
                high = removals - 1

        return max_removable