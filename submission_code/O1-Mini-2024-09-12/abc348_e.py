import sys
import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    from collections import defaultdict
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A = int(data[idx]); idx +=1
        B = int(data[idx]); idx +=1
        edges[A].append(B)
        edges[B].append(A)
    C = [0] + [int(x) for x in data[idx:idx+N]]
    
    sumC = [0]*(N+1)
    f = [0]*(N+1)
    total = sum(C)
    
    def dfs1(u, parent, depth):
        sumC[u] = C[u]
        f[1] += C[u] * depth
        for v in edges[u]:
            if v != parent:
                dfs1(v, u, depth+1)
                sumC[u] += sumC[v]
    
    dfs1(1, -1, 0)
    
    min_f = f[1]
    
    def dfs2(u, parent):
        nonlocal min_f
        for v in edges[u]:
            if v != parent:
                f[v] = f[u] + total - 2*sumC[v]
                if f[v] < min_f:
                    min_f = f[v]
                dfs2(v, u)
    
    dfs2(1, -1)
    print(min_f)

if __name__ == "__main__":
    main()