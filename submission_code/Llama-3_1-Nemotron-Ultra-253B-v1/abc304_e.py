import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    M = int(data[idx])
    idx +=1
    
    parent = list(range(N+1))
    size = [1]*(N+1)
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        if size[u_root] < size[v_root]:
            u_root, v_root = v_root, u_root
        parent[v_root] = u_root
        size[u_root] += size[v_root]
    
    for _ in range(M):
        u = int(data[idx])
        idx +=1
        v = int(data[idx])
        idx +=1
        if u != v:
            union(u, v)
    
    K = int(data[idx])
    idx +=1
    
    forbidden = set()
    for _ in range(K):
        x = int(data[idx])
        idx +=1
        y = int(data[idx])
        idx +=1
        rx = find(x)
        ry = find(y)
        if rx > ry:
            rx, ry = ry, rx
        forbidden.add((rx, ry))
    
    Q = int(data[idx])
    idx +=1
    
    res = []
    for _ in range(Q):
        p = int(data[idx])
        idx +=1
        q = int(data[idx])
        idx +=1
        a = find(p)
        b = find(q)
        if a == b:
            res.append("Yes")
        else:
            if a > b:
                a, b = b, a
            if (a, b) in forbidden:
                res.append("No")
            else:
                res.append("Yes")
    
    print('
'.join(res))

if __name__ == '__main__':
    main()