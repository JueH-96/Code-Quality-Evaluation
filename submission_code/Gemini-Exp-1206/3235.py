class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        dist = [[float('inf')] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0
        for i in range(len(original)):
            o = ord(original[i]) - ord('a')
            c = ord(changed[i]) - ord('a')
            dist[o][c] = min(dist[o][c], cost[i])
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        ans = 0
        for i in range(len(source)):
            if source[i] == target[i]:
                continue
            s = ord(source[i]) - ord('a')
            t = ord(target[i]) - ord('a')
            if dist[s][t] == float('inf'):
                return -1
            ans += dist[s][t]
        return ans