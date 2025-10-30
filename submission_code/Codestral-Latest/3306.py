class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        unmarked_sum = sum(nums)
        result = []

        for index_i, k_i in queries:
            if not marked[index_i]:
                marked[index_i] = True
                unmarked_sum -= nums[index_i]

            count = 0
            for j in range(n):
                if not marked[j]:
                    marked[j] = True
                    unmarked_sum -= nums[j]
                    count += 1
                    if count == k_i:
                        break

            result.append(unmarked_sum)

        return result