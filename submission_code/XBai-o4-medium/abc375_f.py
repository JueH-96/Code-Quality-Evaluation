import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1

    adj = [[] for _ in range(N+1)]  # 1-based indexing

    for road_idx in range(M):
        A = int(input[ptr]); ptr +=1
        B = int(input[ptr]); ptr +=1
        C = int(input[ptr]); ptr +=1
        adj[A].append( (B, C, road_idx) )
        adj[B].append( (A, C, road_idx) )

    closed = [False] * M

    output = []
    INF = 1 << 60

    for _ in range(Q):
        type_q = int(input[ptr]); ptr +=1
        if type_q == 1:
            i = int(input[ptr]); ptr +=1
            closed[i-1] = True
        else:
            x = int(input[ptr]); ptr +=1
            y = int(input[ptr]); ptr +=1
            dist = [INF] * (N+1)
            dist[x] = 0
            heap = [ (0, x) ]
            found = False
            while heap:
                d, u = heapq.heappop(heap)
                if u == y:
                    output.append(str(d))
                    found = True
                    break
                if d > dist[u]:
                    continue
                for v, cost, road_idx in adj[u]:
                    if closed[road_idx]:
                        continue
                    if dist[v] > d + cost:
                        dist[v] = d + cost
                        heapq.heappush(heap, (dist[v], v))
            if not found:
                output.append("-1")
    
    print('
'.join(output))

if __name__ == "__main__":
    main()