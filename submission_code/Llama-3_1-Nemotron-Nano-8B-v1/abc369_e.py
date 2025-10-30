import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    
    adj = [[] for _ in range(N+1)]
    for idx in range(1, M+1):
        u = int(input[ptr])
        ptr +=1
        v = int(input[ptr])
        ptr +=1
        t = int(input[ptr])
        ptr +=1
        adj[u].append( (v, t, idx) )
        adj[v].append( (u, t, idx) )
    
    Q = int(input[ptr])
    ptr +=1
    for _ in range(Q):
        K = int(input[ptr])
        ptr +=1
        B_list = list(map(int, input[ptr:ptr+K]))
        ptr +=K
        
        required_bridges = set(B_list)
        pos_map = {b:i for i, b in enumerate(B_list)}
        required_mask = (1 << K) -1
        
        INF = float('inf')
        dist = [ [INF] * (1 << K) for _ in range(N+1) ]
        dist[1][0] = 0
        heap = []
        heapq.heappush(heap, (0, 1, 0))
        found = False
        
        while heap:
            time, u, mask = heapq.heappop(heap)
            if u == N and mask == required_mask:
                print(time)
                found = True
                break
            if time > dist[u][mask]:
                continue
            for v, t, idx in adj[u]:
                new_mask = mask
                if idx in required_bridges:
                    new_mask = mask | (1 << pos_map[idx])
                new_time = time + t
                if new_time < dist[v][new_mask]:
                    dist[v][new_mask] = new_time
                    heapq.heappush(heap, (new_time, v, new_mask))
        if not found:
            print(-1)
    
if __name__ == '__main__':
    main()