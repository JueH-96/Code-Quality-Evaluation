import sys
sys.setrecursionlimit(1 << 25)

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    
    parent = list(range(N + 1))
    rank = [1] * (N + 1)
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        # Union by rank
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[v_root] = u_root
            if rank[u_root] == rank[v_root]:
                rank[u_root] += 1
    
    for _ in range(M):
        u, v = map(int, input().split())
        union(u, v)
    
    K = int(input())
    forbidden = set()
    for _ in range(K):
        x, y = map(int, input().split())
        rx = find(x)
        ry = find(y)
        if rx != ry:
            a, b = sorted([rx, ry])
            forbidden.add((a, b))
    
    Q = int(input())
    for _ in range(Q):
        p, q = map(int, input().split())
        rp = find(p)
        rq = find(q)
        if rp == rq:
            print("Yes")
        else:
            a = min(rp, rq)
            b = max(rp, rq)
            if (a, b) in forbidden:
                print("No")
            else:
                print("Yes")

if __name__ == "__main__":
    main()