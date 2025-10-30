from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # there are only 26 letters
        INF = float('inf')
        # initialize distance matrix for graph of letters 'a' to 'z'
        dist = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0

        # record the allowed transitions: update using min cost for same edge possible multiples times
        m = len(original)
        for i in range(m):
            u = ord(original[i]) - ord('a')
            v = ord(changed[i]) - ord('a')
            w = cost[i]
            if w < dist[u][v]:
                dist[u][v] = w

        # Floyd-Warshall: find the minimum cost to get from letter x to letter y via any intermediate transformations.
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        # Now, compute the cost for each character in source to turn it into corresponding one in target.
        total_cost = 0
        n = len(source)
        for i in range(n):
            s_char = source[i]
            t_char = target[i]
            if s_char == t_char:
                continue
            u = ord(s_char) - ord('a')
            v = ord(t_char) - ord('a')
            # if there is no sequence of operations to convert s_char to t_char, return -1
            if dist[u][v] == INF:
                return -1
            total_cost += dist[u][v]
        return total_cost