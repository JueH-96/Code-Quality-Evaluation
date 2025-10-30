import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]
    roads = []
    for _ in range(M):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        roads.append((A, B, C))
        adj[A].append((B, C))
        adj[B].append((A, C))
    
    def dijkstra(start):
        INF = float('inf')
        dist = [INF] * (N + 1)
        cnt = [0] * (N + 1)
        dist[start] = 0
        cnt[start] = 1
        heap = []
        heapq.heappush(heap, (0, start))
        
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, c in adj[u]:
                if dist[v] > d + c:
                    dist[v] = d + c
                    cnt[v] = cnt[u]
                    heapq.heappush(heap, (dist[v], v))
                elif dist[v] == d + c:
                    cnt[v] += cnt[u]
        return dist, cnt
    
    dist1, cnt1 = dijkstra(1)
    distN, cntN = dijkstra(N)
    
    D = dist1[N]
    
    for A, B, C in roads:
        cond1 = (dist1[A] + C + distN[B] == D)
        cond2 = (dist1[B] + C + distN[A] == D)
        if not (cond1 or cond2):
            print("No")
        else:
            term1 = cnt1[A] * cntN[B] if cond1 else 0
            term2 = cnt1[B] * cntN[A] if cond2 else 0
            sum_paths = term1 + term2
            if sum_paths == cnt1[N]:
                print("Yes")
            else:
                print("No")

if __name__ == '__main__':
    main()