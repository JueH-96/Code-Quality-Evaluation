from typing import List
from bisect import bisect_left, insort

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        max_and_sums, result, sorted_queries, n = [], [], sorted(zip(queries, range(len(queries)))), len(queries)

        for num1, num2 in zip(nums1, nums2):
            index_to_insert = bisect_left(max_and_sums, num2 - num1)
            insort(max_and_sums, num1 + num2)
            if index_to_insert < len(max_and_sums):
                max_and_sums[index_to_insert] += num2 - num1

        for x, y in sorted(zip(*sorted(x for x, _ in sorted_queries[::1]), key=lambda x: (x[0], -x[1])))[:n]:
            current_query_index = sorted_queries.pop()[1]
            min_required = x + y
            result.append(max_and_sums[bisect_left(max_and_sums, min_required)] if max_and_sums[bisect_left(max_and_sums, min_required)] >= min_required else -1)

        return result[::-1]