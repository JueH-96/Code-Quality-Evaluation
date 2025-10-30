import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    X = int(input[ptr]); ptr +=1

    forward_adj = [[] for _ in range(N+1)]
    backward_adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(input[ptr]); ptr +=1
        v = int(input[ptr]); ptr +=1
        forward_adj[u].append(v)
        backward_adj[v].append(u)

    INF = 10**18
    # dist[0][u] is the distance to node u with 0 flips (original direction)
    # dist[1][u] is the distance to node u with 1 flip (reversed direction)
    dist = [[INF] * (N+1) for _ in range(2)]
    dist[0][1] = 0

    heap = []
    heapq.heappush(heap, (0, 1, 0))

    while heap:
        current_cost, u, f = heapq.heappop(heap)
        if u == N:
            print(current_cost)
            return
        if current_cost > dist[f][u]:
            continue
        # Process moving along the current direction
        if f == 0:
            neighbors = forward_adj[u]
        else:
            neighbors = backward_adj[u]
        for v in neighbors:
            if dist[f][v] > current_cost + 1:
                dist[f][v] = current_cost + 1
                heapq.heappush(heap, (dist[f][v], v, f))
        # Process reversing the edges
        new_f = 1 - f
        new_cost = current_cost + X
        if dist[new_f][u] > new_cost:
            dist[new_f][u] = new_cost
            heapq.heappush(heap, (new_cost, u, new_f))

if __name__ == "__main__":
    main()