class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        # Precompute dp and prefix_max
        dp = [[] for _ in range(n)]
        prefix_max = [[] for _ in range(n)]
        for a in reversed(range(n)):
            max_k_a = n - 1 - a
            dp_a = [0] * (max_k_a + 1)
            pm_a = [0] * (max_k_a + 1)
            dp_a[0] = nums[a]
            pm_a[0] = nums[a]
            for k in range(1, max_k_a + 1):
                lsb = k & -k
                prev_k = k ^ lsb
                new_a = a + lsb
                if new_a >= n:
                    # This case should not occur due to max_k_a's definition
                    dp_a_k = 0
                else:
                    if prev_k >= len(dp[new_a]):
                        # Should not happen since new_a's max_k is n-1 - new_a >= prev_k
                        dp_a_k = 0
                    else:
                        dp_a_k = dp_a[prev_k] ^ dp[new_a][prev_k]
                dp_a[k] = dp_a_k
                pm_a[k] = max(pm_a[k-1], dp_a[k])
            dp[a] = dp_a
            prefix_max[a] = pm_a
        # Process queries
        answer = []
        for L, R in queries:
            max_xor = 0
            for a in range(L, R + 1):
                k = R - a
                if k < 0:
                    continue
                if k >= len(prefix_max[a]):
                    continue
                current = prefix_max[a][k]
                if current > max_xor:
                    max_xor = current
            answer.append(max_xor)
        return answer