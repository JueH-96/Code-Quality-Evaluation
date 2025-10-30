class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        n = len(nums)
        arr = sorted([(nums[i], cost1 if i == j else cost2) for i in range(n) for j in range(n)])
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + arr[i - 1][1]
        res = 0
        for i in range(1, n):
            if arr[i][0] == arr[i - 1][0]:
                res += min(arr[i - 1][1], arr[i][1]) * (arr[i][0] - arr[i - 1][0])
                arr[i][1] = min(arr[i - 1][1], arr[i][1])
        for i in range(n):
            res += (arr[i][0] - 1) * arr[i][1]
        return res % (10**9 + 7)