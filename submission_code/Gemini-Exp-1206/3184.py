class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        arr = sorted([[nums[i] - i, i] for i in range(n)])
        
        bit = [0] * (n + 1)

        def update(i, val):
            while i <= n:
                bit[i] = max(bit[i], val)
                i += i & -i

        def query(i):
            res = -float('inf')
            while i > 0:
                res = max(res, bit[i])
                i -= i & -i
            return res

        rank = {}
        for i in range(n):
            rank[arr[i][1]] = i + 1

        ans = -float('inf')
        for i in range(n):
            num = nums[i]
            idx = i
            prev_max = query(rank[idx])
            if prev_max == -float('inf'):
                update(rank[idx], num)
                ans = max(ans, num)
            else:
                update(rank[idx], prev_max + num)
                ans = max(ans, prev_max + num)

        return ans