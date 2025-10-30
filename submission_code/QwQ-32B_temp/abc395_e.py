import sys
import heapq

def main():
    sys.setrecursionlimit(1 << 25)
    N, M, X = map(int, sys.stdin.readline().split())
    adj_forward = [[] for _ in range(N + 1)]
    adj_backward = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj_forward[u].append(v)
        adj_backward[v].append(u)
    
    INF = 10**18
    dist0 = [INF] * (N + 1)  # direction 0
    dist1 = [INF] * (N + 1)  # direction 1
    dist0[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1, 0))
    
    while heap:
        current_dist, u, current_dir = heapq.heappop(heap)
        if u == N:
            print(current_dist)
            return
        if current_dir == 0:
            if current_dist > dist0[u]:
                continue
        else:
            if current_dist > dist1[u]:
                continue
        
        # Process moving edges
        if current_dir == 0:
            adj_list = adj_forward[u]
        else:
            adj_list = adj_backward[u]
        for v in adj_list:
            new_dist = current_dist + 1
            if current_dir == 0:
                if new_dist < dist0[v]:
                    dist0[v] = new_dist
                    heapq.heappush(heap, (new_dist, v, 0))
            else:
                if new_dist < dist1[v]:
                    dist1[v] = new_dist
                    heapq.heappush(heap, (new_dist, v, 1))
        
        # Process reversal
        new_dir = 1 - current_dir
        new_dist_rev = current_dist + X
        if new_dir == 0:
            if new_dist_rev < dist0[u]:
                dist0[u] = new_dist_rev
                heapq.heappush(heap, (new_dist_rev, u, 0))
        else:
            if new_dist_rev < dist1[u]:
                dist1[u] = new_dist_rev
                heapq.heappush(heap, (new_dist_rev, u, 1))
    
    # The problem states that a path exists, so this line shouldn't be reached
    ans = min(dist0[N], dist1[N])
    print(ans)

if __name__ == "__main__":
    main()