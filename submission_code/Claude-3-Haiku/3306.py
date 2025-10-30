class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        marked = [False] * n
        answer = []

        for index, k in queries:
            marked[index] = True
            unmarked = sorted([i for i, x in enumerate(nums) if not marked[i]])
            for i in range(min(k, len(unmarked))):
                marked[unmarked[i]] = True

            unmarked_sum = sum(nums[i] for i in range(n) if not marked[i])
            answer.append(unmarked_sum)

        return answer