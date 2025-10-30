# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.readline
    INF = float('inf')
    
    # Read inputs
    N, A, B, C = map(int, input().split())
    D = [list(map(int, input().split())) for _ in range(N)]
    
    # Dijkstra-like function for "car only" on a complete graph of size N
    def dijkstra_car(start):
        dist = [INF]*N
        dist[start] = 0
        visited = [False]*N
        for _ in range(N):
            u = -1
            u_dist = INF
            # Pick the unvisited node with the smallest dist
            for i in range(N):
                if not visited[i] and dist[i] < u_dist:
                    u_dist = dist[i]
                    u = i
            if u == -1:
                break
            visited[u] = True
            # Relax distances using the cost for car
            for v in range(N):
                if not visited[v]:
                    alt = u_dist + D[u][v] * A
                    if alt < dist[v]:
                        dist[v] = alt
        return dist
    
    # Dijkstra-like function for "train only" on a complete graph of size N
    def dijkstra_train(start):
        dist = [INF]*N
        dist[start] = 0
        visited = [False]*N
        for _ in range(N):
            u = -1
            u_dist = INF
            # Pick the unvisited node with the smallest dist
            for i in range(N):
                if not visited[i] and dist[i] < u_dist:
                    u_dist = dist[i]
                    u = i
            if u == -1:
                break
            visited[u] = True
            # Relax distances using the cost for train
            for v in range(N):
                if not visited[v]:
                    alt = u_dist + D[u][v] * B + C
                    if alt < dist[v]:
                        dist[v] = alt
        return dist
    
    # Compute the minimum cost to reach each city by car (from city 1 => index 0)
    cost_car = dijkstra_car(0)
    # Compute the minimum cost to reach city N (index N-1) by train from each city
    # because the graph is undirected, we can just "start" from N-1
    cost_train = dijkstra_train(N-1)
    
    # Either we don't switch at all (cost_car[N-1]), or we switch in some city i
    # total cost = cost_car[i] + cost_train[i].
    ans = cost_car[N-1]
    for i in range(N):
        cur = cost_car[i] + cost_train[i]
        if cur < ans:
            ans = cur
    
    print(ans)

if __name__ == "__main__":
    main()