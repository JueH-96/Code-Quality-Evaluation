import sys
from itertools import permutations

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

bridges = []
for i in range(M):
    U = int(data[index])
    index += 1
    V = int(data[index])
    index += 1
    T = int(data[index])
    index += 1
    bridges.append((U, V, T))

Q = int(data[index])
index += 1

queries = []
for i in range(Q):
    K = int(data[index])
    index += 1
    B = [int(data[index + j]) for j in range(K)]
    index += K
    queries.append((K, B))

def calculate_min_time(bridges, query):
    K, B = query
    min_time = float('inf')

    for perm in permutations(B):
        current_time = 0
        visited = set()
        for bridge_index in perm:
            U, V, T = bridges[bridge_index - 1]
            current_time += T
            visited.add(U)
            visited.add(V)

        if 1 in visited and N in visited:
            min_time = min(min_time, current_time)

    return min_time

results = []
for query in queries:
    result = calculate_min_time(bridges, query)
    results.append(result)

for result in results:
    print(result)