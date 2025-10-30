# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    testimonies = []
    index = 2
    for _ in range(M):
        A = int(data[index]) - 1
        B = int(data[index + 1]) - 1
        C = int(data[index + 2])
        testimonies.append((A, B, C))
        index += 3
    
    # We will use a Union-Find data structure to manage the relationships
    parent = list(range(2 * N))
    rank = [0] * (2 * N)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1
    
    for A, B, C in testimonies:
        if C == 0:
            union(A, B)
            union(A + N, B + N)
        else:
            union(A, B + N)
            union(A + N, B)
    
    for i in range(N):
        if find(i) == find(i + N):
            print(-1)
            return
    
    result = ['0'] * N
    for i in range(N):
        if find(i) == find(0 + N):
            result[i] = '1'
    
    print("".join(result))

if __name__ == "__main__":
    solve()