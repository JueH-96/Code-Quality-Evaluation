class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        marked = [False] * n
        answer = []

        for index_i, k_i in queries:
            if not marked[index_i]:
                marked[index_i] = True

            unmarked_indices = []
            for i in range(n):
                if not marked[i]:
                    unmarked_indices.append(i)

            unmarked_indices.sort(key=lambda x: (nums[x], x))

            for i in range(min(k_i, len(unmarked_indices))):
                marked[unmarked_indices[i]] = True

            current_sum = 0
            for i in range(n):
                if not marked[i]:
                    current_sum += nums[i]
            answer.append(current_sum)

        return answer