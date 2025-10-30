class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        q = len(queries)

        for k in range(q + 1):
            temp_nums = nums[:]
            for i in range(k):
                l, r, val = queries[i]
                for j in range(l, r + 1):
                    temp_nums[j] -= val

            all_zero = True
            for num in temp_nums:
                if num != 0:
                    all_zero = False
                    break

            if all_zero:
                return k

        return -1