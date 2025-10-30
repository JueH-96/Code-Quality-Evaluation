class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        unmarked_sum = sum(nums)
        marked_sum = 0
        res = []

        for index, k in queries:
            if not marked[index]:
                marked[index] = True
                unmarked_sum -= nums[index]
                marked_sum += nums[index]

            k = min(k, unmarked_sum)
            unmarked_indices = [i for i in range(n) if not marked[i]]
            unmarked_indices.sort()
            for i in unmarked_indices[:k]:
                marked[i] = True
                marked_sum += nums[i]
                unmarked_sum -= nums[i]

            res.append(unmarked_sum)

        return res