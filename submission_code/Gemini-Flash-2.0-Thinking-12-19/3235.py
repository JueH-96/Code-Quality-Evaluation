class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        costs = [[float('inf')] * 26 for _ in range(26)]
        for i in range(26):
            costs[i][i] = 0

        for i in range(len(original)):
            u = ord(original[i]) - ord('a')
            v = ord(changed[i]) - ord('a')
            costs[u][v] = min(costs[u][v], cost[i])

        for k in range(26):
            for i in range(26):
                for j in range(26):
                    costs[i][j] = min(costs[i][j], costs[i][k] + costs[k][j])

        total_cost = 0
        for i in range(n):
            if source[i] != target[i]:
                u = ord(source[i]) - ord('a')
                v = ord(target[i]) - ord('a')
                if costs[u][v] == float('inf'):
                    return -1
                total_cost += costs[u][v]
        return total_cost