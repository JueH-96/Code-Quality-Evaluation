class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        def can_make_zero(current_queries, current_nums):
            n = len(current_nums)
            coverage_count = [0] * n
            for l, r in current_queries:
                for i in range(l, r + 1):
                    coverage_count[i] += 1
            for i in range(n):
                if coverage_count[i] < current_nums[i]:
                    return False
            return True

        if not can_make_zero(queries, nums):
            return -1

        current_queries = list(queries)
        removed_count = 0
        indices_to_remove = list(range(len(queries)))

        for query_index_to_try in range(len(queries)):
            temp_queries = []
            removed_query = None
            original_query_index = -1
            
            temp_queries_list = []
            for i in range(len(current_queries)):
                if i != query_index_to_try:
                    temp_queries_list.append(current_queries[i])
                else:
                    removed_query = current_queries[i]
                    original_query_index = i
            
            if can_make_zero(temp_queries_list, nums):
                current_queries = temp_queries_list
                removed_count += 1
                queries[query_index_to_try] = None # Mark as removed, not actually removing from original queries list

        return removed_count