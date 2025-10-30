class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        pairs = sorted(zip(nums1, nums2), reverse=True)
        indexed_queries = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], reverse=True)
        answers = [-1] * len(queries)
        relevant_sums = []  # Stores (nums2, sum)
        pair_idx = 0

        for x, y, query_index in indexed_queries:
            # Add relevant pairs to the list
            while pair_idx < n and pairs[pair_idx][0] >= x:
                relevant_sums.append((pairs[pair_idx][1], pairs[pair_idx][0] + pairs[pair_idx][1]))
                pair_idx += 1

            max_sum = -1
            for r_nums2, r_sum in relevant_sums:
                if r_nums2 >= y:
                    max_sum = max(max_sum, r_sum)
            answers[query_index] = max_sum

        return answers