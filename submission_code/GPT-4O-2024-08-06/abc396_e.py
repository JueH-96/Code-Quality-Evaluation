def find_minimal_good_sequence(N, M, constraints):
    from collections import defaultdict, deque
    
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    value = [0] * (N + 1)
    graph = defaultdict(list)
    
    def find(x):
        if parent[x] != x:
            orig = parent[x]
            parent[x] = find(parent[x])
            value[x] ^= value[orig]
        return parent[x]
    
    def union(x, y, z):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
                value[rootY] = value[x] ^ value[y] ^ z
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
                value[rootX] = value[x] ^ value[y] ^ z
            else:
                parent[rootY] = rootX
                value[rootY] = value[x] ^ value[y] ^ z
                rank[rootX] += 1
        else:
            if (value[x] ^ value[y]) != z:
                return False
        return True
    
    for x, y, z in constraints:
        if not union(x, y, z):
            return -1
    
    result = [0] * N
    for i in range(1, N + 1):
        find(i)
        result[i - 1] = value[i]
    
    return result

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
constraints = []

index = 2
for _ in range(M):
    X = int(data[index])
    Y = int(data[index + 1])
    Z = int(data[index + 2])
    constraints.append((X, Y, Z))
    index += 3

result = find_minimal_good_sequence(N, M, constraints)
if result == -1:
    print(-1)
else:
    print(" ".join(map(str, result)))