class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        nums = sorted((num, i) for i, num in enumerate(nums))
        queries = sorted((q[1], q[0], i) for i, q in enumerate(queries))
        ans = [0] * len(queries)
        bit = [0] * (len(nums) + 1)

        def update(i, val):
            while i <= len(nums):
                bit[i] = max(bit[i], val)
                i += i & -i

        def query(i):
            res = 0
            while i:
                res = max(res, bit[i])
                i -= i & -i
            return res

        j = 0
        for q in queries:
            while j < len(nums) and nums[j][0] <= q[0]:
                update(nums[j][1] + 1, ans[q[2]])
                ans[q[2]] = max(ans[q[2]], query(nums[j][1]) + nums[j][0])
                j += 1

        return sum(ans) % MOD