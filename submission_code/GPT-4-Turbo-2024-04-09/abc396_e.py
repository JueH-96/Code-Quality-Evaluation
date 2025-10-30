import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    constraints = []
    index = 2
    for _ in range(M):
        X = int(data[index]) - 1
        Y = int(data[index + 1]) - 1
        Z = int(data[index + 2])
        constraints.append((X, Y, Z))
        index += 3
    
    # We will use a Union-Find structure with an additional map to store the XOR differences
    parent = list(range(N))
    rank = [0] * N
    xor_to_parent = [0] * N  # xor_to_parent[i] is the XOR of node i with its parent
    
    def find(x):
        if parent[x] != x:
            original_parent = parent[x]
            parent[x] = find(parent[x])
            xor_to_parent[x] ^= xor_to_parent[original_parent]
        return parent[x]
    
    def union(x, y, val):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
                xor_to_parent[root_y] = xor_to_parent[x] ^ xor_to_parent[y] ^ val
            elif rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
                xor_to_parent[root_x] = xor_to_parent[x] ^ xor_to_parent[y] ^ val
            else:
                parent[root_y] = root_x
                xor_to_parent[root_y] = xor_to_parent[x] ^ xor_to_parent[y] ^ val
                rank[root_x] += 1
        else:
            # Same root, check for consistency
            if (xor_to_parent[x] ^ xor_to_parent[y]) != val:
                return False
        return True
    
    # Process each constraint
    for x, y, z in constraints:
        if not union(x, y, z):
            print(-1)
            return
    
    # If we reach here, a solution exists. We need to find the sequence A that minimizes the sum.
    # We can set one representative for each connected component to 0 to minimize the sum.
    A = [0] * N
    for i in range(N):
        A[i] = xor_to_parent[i] ^ A[find(i)]
    
    print(" ".join(map(str, A)))