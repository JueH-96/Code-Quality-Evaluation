import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    k = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        edges.append((u, v, w))
    
    A = [int(next(it)) for _ in range(k)]
    B = [int(next(it)) for _ in range(k)]
    
    freqA = [0] * (n + 1)
    for a_node in A:
        freqA[a_node] += 1
            
    freqB = [0] * (n + 1)
    for b_node in B:
        freqB[b_node] += 1
        
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    a_arr = [0] * (n + 1)
    b_arr = [0] * (n + 1)
    min_ab_arr = [0] * (n + 1)
    
    for i in range(1, n + 1):
        a_arr[i] = freqA[i]
        b_arr[i] = freqB[i]
        min_ab_arr[i] = min(freqA[i], freqB[i])
    
    total_min = sum(min_ab_arr[1:])
    
    edges.sort(key=lambda x: x[2])
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    ans = 0
    prev = 0
    i = 0
    while i < m:
        w = edges[i][2]
        ans += (w - prev) * (k - total_min)
        
        j = i
        while j < m and edges[j][2] == w:
            u, v, _ = edges[j]
            j += 1
            ru = find(u)
            rv = find(v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            total_min -= min_ab_arr[ru] + min_ab_arr[rv]
            a_arr[ru] += a_arr[rv]
            b_arr[ru] += b_arr[rv]
            min_ab_arr[ru] = min(a_arr[ru], b_arr[ru])
            total_min += min_ab_arr[ru]
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
        i = j
        prev = w
    
    print(ans)

if __name__ == '__main__':
    main()