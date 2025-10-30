import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    m = int(data[1])
    parent = [i for i in range(n+1)]
    size_arr = [1] * (n+1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        a = find(a)
        b = find(b)
        if a == b:
            return
        if size_arr[a] < size_arr[b]:
            a, b = b, a
        parent[b] = a
        size_arr[a] += size_arr[b]

    idx = 2
    for _ in range(m):
        u = int(data[idx])
        v = int(data[idx+1])
        idx += 2
        union(u, v)
        
    roots = set()
    for i in range(1, n+1):
        roots.add(find(i))
        
    total_edges = 0
    for r in roots:
        s = size_arr[r]
        total_edges += s * (s - 1) // 2
        
    ans = total_edges - m
    print(ans)

if __name__ == '__main__':
    main()