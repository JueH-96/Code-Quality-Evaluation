class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)

        def check(removed_queries):
            temp_nums = nums[:]
            used_queries = []
            for i in range(m):
                if i not in removed_queries:
                    used_queries.append(queries[i])

            if not used_queries:
                return all(x == 0 for x in temp_nums)

            diff = [0] * (n + 1)
            for l, r in used_queries:
                diff[l] += 1
                diff[r + 1] -= 1

            for i in range(n):
                if i > 0:
                    diff[i] += diff[i - 1]

            for i in range(n):
                if diff[i] < temp_nums[i]:
                    return False
            return True

        max_removed = -1
        for i in range(1 << m):
            removed_queries = []
            for j in range(m):
                if (i >> j) & 1:
                    removed_queries.append(j)

            if check(removed_queries):
                max_removed = max(max_removed, len(removed_queries))

        return max_removed