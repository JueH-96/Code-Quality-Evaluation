import heapq
import sys

mod1 = 10**9 + 7
mod2 = 998244353

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    edges = []
    graph = [[] for _ in range(n + 1)]
    index = 2
    for i in range(m):
        a = int(data[index])
        b = int(data[index + 1])
        c = int(data[index + 2])
        index += 3
        edges.append((a, b, c))
        graph[a].append((b, c))
        graph[b].append((a, c))
    
    INF = 10**18
    dist1 = [INF] * (n + 1)
    dist1[1] = 0
    heap = [(0, 1)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist1[u]:
            continue
        for (v, c) in graph[u]:
            nd = d + c
            if nd < dist1[v]:
                dist1[v] = nd
                heapq.heappush(heap, (nd, v))
    
    dp1_mod1 = [0] * (n + 1)
    dp1_mod2 = [0] * (n + 1)
    dp1_mod1[1] = 1
    dp1_mod2[1] = 1
    
    nodes = list(range(1, n + 1))
    nodes.sort(key=lambda x: dist1[x])
    
    for u in nodes:
        for (v, c) in graph[u]:
            if dist1[u] + c == dist1[v]:
                dp1_mod1[v] = (dp1_mod1[v] + dp1_mod1[u]) % mod1
                dp1_mod2[v] = (dp1_mod2[v] + dp1_mod2[u]) % mod2
    
    total_mod1 = dp1_mod1[n]
    total_mod2 = dp1_mod2[n]
    
    dp2_mod1 = [0] * (n + 1)
    dp2_mod2 = [0] * (n + 1)
    dp2_mod1[n] = 1
    dp2_mod2[n] = 1
    
    nodes_desc = sorted(nodes, key=lambda x: dist1[x], reverse=True)
    
    for u in nodes_desc:
        for (v, c) in graph[u]:
            if dist1[u] + c == dist1[v]:
                dp2_mod1[u] = (dp2_mod1[u] + dp2_mod1[v]) % mod1
                dp2_mod2[u] = (dp2_mod2[u] + dp2_mod2[v]) % mod2
    
    out_lines = []
    for (a, b, c) in edges:
        if dist1[a] + c == dist1[b]:
            count1 = (dp1_mod1[a] * dp2_mod1[b]) % mod1
            count2 = (dp1_mod2[a] * dp2_mod2[b]) % mod2
            if count1 == total_mod1 and count2 == total_mod2:
                out_lines.append("Yes")
            else:
                out_lines.append("No")
        elif dist1[b] + c == dist1[a]:
            count1 = (dp1_mod1[b] * dp2_mod1[a]) % mod1
            count2 = (dp1_mod2[b] * dp2_mod2[a]) % mod2
            if count1 == total_mod1 and count2 == total_mod2:
                out_lines.append("Yes")
            else:
                out_lines.append("No")
        else:
            out_lines.append("No")
    
    print("
".join(out_lines))

if __name__ == '__main__':
    main()