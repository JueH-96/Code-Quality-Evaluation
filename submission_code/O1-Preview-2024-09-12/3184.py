class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        class BIT:
            def __init__(self, size):
                self.N = size + 2  # Plus extra to avoid index issues
                self.tree = [float('-inf')] * self.N

            def update(self, idx, val):
                idx += 1
                while idx < self.N:
                    self.tree[idx] = max(self.tree[idx], val)
                    idx += idx & -idx

            def query(self, idx):
                res = float('-inf')
                idx += 1
                while idx > 0:
                    res = max(res, self.tree[idx])
                    idx -= idx & -idx
                return res

        N = len(nums)
        D_list = [nums[i] - i for i in range(N)]
        unique_D = sorted(set(D_list))
        D_to_compressed = {d: idx for idx, d in enumerate(unique_D)}

        bit = BIT(len(unique_D))
        dp = [0] * N
        max_sum = float('-inf')
        for i in range(N):
            D_i = D_list[i]
            compressed_D_i = D_to_compressed[D_i]
            max_dp_before = bit.query(compressed_D_i)  # Get max dp[j] where D_j <= D_i
            dp[i] = nums[i]
            if max_dp_before != float('-inf'):
                dp[i] = max(dp[i], max_dp_before + nums[i])
            bit.update(compressed_D_i, dp[i])
            max_sum = max(max_sum, dp[i])

        return max_sum