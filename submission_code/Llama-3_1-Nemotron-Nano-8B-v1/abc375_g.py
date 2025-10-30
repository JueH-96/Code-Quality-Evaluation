import heapq
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    edges = []
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(input[idx]); idx += 1
        b = int(input[idx]); idx += 1
        c = int(input[idx]); idx += 1
        edges.append((a, b, c))
        adj[a].append((b, c))
        adj[b].append((a, c))
    
    def dijkstra(start):
        dist = [float('inf')] * (N + 1)
        cnt = [0] * (N + 1)
        dist[start] = 0
        cnt[start] = 1
        heap = [(0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if dist[v] > d + w:
                    dist[v] = d + w
                    cnt[v] = cnt[u]
                    heapq.heappush(heap, (dist[v], v))
                elif dist[v] == d + w:
                    cnt[v] += cnt[u]
        return dist, cnt
    
    dist1, cnt1 = dijkstra(1)
    distn, cntn = dijkstra(N)
    D = dist1[N]
    cnt_total = cnt1[N]
    
    for a, b, c in edges:
        if (dist1[a] + c + distn[b] == D) or (dist1[b] + c + distn[a] == D):
            contrib = 0
            if (dist1[a] + c == dist1[b]) and (distn[b] + c == distn[a]):
                contrib += cnt1[a] * cntn[b]
            if (dist1[b] + c == dist1[a]) and (distn[a] + c == distn[b]):
                contrib += cnt1[b] * cntn[a]
            print("Yes" if contrib == cnt_total else "No")
        else:
            print("No")

if __name__ == "__main__":
    main()