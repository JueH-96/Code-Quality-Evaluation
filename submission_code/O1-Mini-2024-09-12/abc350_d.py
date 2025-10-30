# YOUR CODE HERE
import sys
import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    edges = data[2:]
    parent = list(range(N+1))
    size = [1]*(N+1)
    edge_count = [0]*(N+1)
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    for i in range(0, 2*M, 2):
        A = int(edges[i])
        B = int(edges[i+1])
        rootA = find(A)
        rootB = find(B)
        if rootA != rootB:
            if size[rootA] < size[rootB]:
                rootA, rootB = rootB, rootA
            parent[rootB] = rootA
            size[rootA] += size[rootB]
            edge_count[rootA] += edge_count[rootB] +1
        else:
            edge_count[rootA] +=1
    
    seen = set()
    total =0
    for i in range(1, N+1):
        root = find(i)
        if root not in seen:
            seen.add(root)
            s = size[root]
            m = edge_count[root]
            total += s*(s-1)//2 - m
    print(total)

if __name__ == "__main__":
    main()