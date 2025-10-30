import sys

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    for i in range(N-1):
        edges.append((int(data[1 + 2*i]), int(data[2 + 2*i])))
    C = list(map(int, data[N*2+1:N*2+1+N]))
    
    adjacency = [[] for _ in range(N)]
    for a, b in edges:
        adjacency[a-1].append(b-1)
        adjacency[b-1].append(a-1)
    
    sumC = [0] * N
    sz = [0] * N
    depth = [0] * N
    f_root = 0
    
    def DFS1(node, parent, depth_val):
        nonlocal f_root
        sz[node] = 1
        sumC[node] = C[node]
        depth[node] = depth_val
        f_root += C[node] * depth_val
        for child in adjacency[node]:
            if child != parent:
                DFS1(child, node, depth_val + 1)
                sz[node] += sz[child]
                sumC[node] += sumC[child]
    
    DFS1(0, -1, 0)
    
    sum_all_C = sum(C)
    f = [0] * N
    min_f = float('inf')
    
    def DFS2(node, parent):
        nonlocal min_f
        if parent != -1:
            f[node] = f[parent] - 2 * sumC[node] + sum_all_C
        else:
            f[node] = f_root
        if f[node] < min_f:
            min_f = f[node]
        for child in adjacency[node]:
            if child != parent:
                DFS2(child, node)
    
    DFS2(0, -1)
    print(min_f)

if __name__ == "__main__":
    main()