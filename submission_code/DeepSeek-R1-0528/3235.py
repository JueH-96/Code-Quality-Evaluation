class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n_letters = 26
        INF = 10**18
        dist = [[INF] * n_letters for _ in range(n_letters)]
        
        for i in range(n_letters):
            dist[i][i] = 0
        
        for o, ch, co in zip(original, changed, cost):
            u = ord(o) - ord('a')
            v = ord(ch) - ord('a')
            if co < dist[u][v]:
                dist[u][v] = co
        
        for k in range(n_letters):
            for i in range(n_letters):
                if dist[i][k] == INF:
                    continue
                for j in range(n_letters):
                    new_cost = dist[i][k] + dist[k][j]
                    if new_cost < dist[i][j]:
                        dist[i][j] = new_cost
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            u = ord(s_char) - ord('a')
            v = ord(t_char) - ord('a')
            if dist[u][v] == INF:
                return -1
            total_cost += dist[u][v]
        
        return total_cost