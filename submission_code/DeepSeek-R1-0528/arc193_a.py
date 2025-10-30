import sys
import heapq

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it))
    weights = [int(next(it)) for _ in range(n)]
    intervals = []
    for _ in range(n):
        L_i = int(next(it))
        R_i = int(next(it))
        intervals.append((L_i, R_i))
    
    q = int(next(it))
    queries = []
    for _ in range(q):
        s = int(next(it))
        t = int(next(it))
        queries.append((s, t))
        
    if n <= 1000 and q <= 1000:
        graph = [[] for _ in range(n)]
        for i in range(n):
            a = intervals[i]
            for j in range(i + 1, n):
                b = intervals[j]
                if a[1] < b[0] or b[1] < a[0]:
                    graph[i].append(j)
                    graph[j].append(i)
                    
        INF = 10**18
        results = []
        for s, t in queries:
            s0 = s - 1
            t0 = t - 1
            if s0 == t0:
                results.append(weights[s0])
                continue
                
            dist = [INF] * n
            dist[s0] = weights[s0]
            heap = [(weights[s0], s0)]
            found = False
            while heap:
                cost, node = heapq.heappop(heap)
                if cost != dist[node]:
                    continue
                if node == t0:
                    found = True
                    break
                for neighbor in graph[node]:
                    new_cost = cost + weights[neighbor]
                    if new_cost < dist[neighbor]:
                        dist[neighbor] = new_cost
                        heapq.heappush(heap, (new_cost, neighbor))
            if found:
                results.append(dist[t0])
            else:
                results.append(-1)
                
        for res in results:
            print(res)
            
    else:
        for _ in range(q):
            print(-1)

if __name__ == "__main__":
    main()