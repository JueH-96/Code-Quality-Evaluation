import heapq
import sys

def main():
    data = sys.stdin.read().split()
    mod1 = 10**9 + 7
    mod2 = 998244353
    
    n = int(data[0])
    m = int(data[1])
    graph = [[] for _ in range(n)]
    edges = []
    index = 2
    for i in range(m):
        a = int(data[index])
        b = int(data[index+1])
        c = int(data[index+2])
        index += 3
        a0 = a - 1
        b0 = b - 1
        graph[a0].append((b0, c))
        graph[b0].append((a0, c))
        edges.append((a0, b0, c))
    
    INF = 10**18
    dist1 = [INF] * n
    dp1_mod1 = [0] * n
    dp1_mod2 = [0] * n
    heap = []
    dist1[0] = 0
    dp1_mod1[0] = 1
    dp1_mod2[0] = 1
    heapq.heappush(heap, (0, 0))
    
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist1[u]:
            continue
        for v, w in graph[u]:
            new_d = d + w
            if new_d < dist1[v]:
                dist1[v] = new_d
                dp1_mod1[v] = dp1_mod1[u]
                dp1_mod2[v] = dp1_mod2[u]
                heapq.heappush(heap, (new_d, v))
            elif new_d == dist1[v]:
                dp1_mod1[v] = (dp1_mod1[v] + dp1_mod1[u]) % mod1
                dp1_mod2[v] = (dp1_mod2[v] + dp1_mod2[u]) % mod2
    
    dist2 = [INF] * n
    dp2_mod1 = [0] * n
    dp2_mod2 = [0] * n
    heap = []
    dist2[n-1] = 0
    dp2_mod1[n-1] = 1
    dp2_mod2[n-1] = 1
    heapq.heappush(heap, (0, n-1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist2[u]:
            continue
        for v, w in graph[u]:
            new_d = d + w
            if new_d < dist2[v]:
                dist2[v] = new_d
                dp2_mod1[v] = dp2_mod1[u]
                dp2_mod2[v] = dp2_mod2[u]
                heapq.heappush(heap, (new_d, v))
            elif new_d == dist2[v]:
                dp2_mod1[v] = (dp2_mod1[v] + dp2_mod1[u]) % mod1
                dp2_mod2[v] = (dp2_mod2[v] + dp2_mod2[u]) % mod2
    
    total_shortest = dist1[n-1]
    total_ways_mod1 = dp1_mod1[n-1]
    total_ways_mod2 = dp1_mod2[n-1]
    
    results = []
    for (a, b, c) in edges:
        cont1 = 0
        cont2 = 0
        if dist1[a] + c + dist2[b] == total_shortest:
            cont1 = (cont1 + dp1_mod1[a] * dp2_mod1[b]) % mod1
            cont2 = (cont2 + dp1_mod2[a] * dp2_mod2[b]) % mod2
        if dist1[b] + c + dist2[a] == total_shortest:
            cont1 = (cont1 + dp1_mod1[b] * dp2_mod1[a]) % mod1
            cont2 = (cont2 + dp1_mod2[b] * dp2_mod2[a]) % mod2
        
        if cont1 == total_ways_mod1 and cont2 == total_ways_mod2:
            results.append("Yes")
        else:
            results.append("No")
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()