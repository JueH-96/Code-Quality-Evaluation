class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = dp[i + 1]
            if source[i] != target[i]:
                min_cost = float('inf')
                for j in range(len(original)):
                    if source[i] == original[j] and target[i] == changed[j]:
                        min_cost = min(min_cost, cost[j])
                if min_cost == float('inf'):
                    return -1
                dp[i] += min_cost
        return dp[0]