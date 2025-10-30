import itertools

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        def can_make_zero(current_nums, current_queries):
            n = len(current_nums)
            coverage_count = [0] * n
            for l, r in current_queries:
                for i in range(l, r + 1):
                    coverage_count[i] += 1
            for i in range(n):
                if coverage_count[i] < current_nums[i]:
                    return False
            return True

        if not can_make_zero(nums, queries):
            return -1

        max_removed_queries = 0
        n_queries = len(queries)
        for k in range(1, n_queries + 1):
            import itertools
            indices_to_remove_combinations = itertools.combinations(range(n_queries), k)
            found_solution_for_k = False
            for indices_to_remove in indices_to_remove_combinations:
                remaining_queries = []
                for i in range(n_queries):
                    if i not in indices_to_remove:
                        remaining_queries.append(queries[i])
                if can_make_zero(nums, remaining_queries):
                    found_solution_for_k = True
                    break
            if found_solution_for_k:
                max_removed_queries = max(max_removed_queries, k)

        return max_removed_queries