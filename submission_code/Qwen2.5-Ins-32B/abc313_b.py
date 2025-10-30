import sys
from collections import defaultdict

def find_strongest_programmer(N, M, relations):
    graph = defaultdict(list)
    in_degree = [0] * (N + 1)
    
    for A, B in relations:
        graph[A].append(B)
        in_degree[B] += 1
    
    strongest_candidates = [i for i in range(1, N + 1) if in_degree[i] == 0]
    
    if len(strongest_candidates) != 1:
        return -1
    
    strongest = strongest_candidates[0]
    
    visited = set()
    stack = [strongest]
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                stack.append(neighbor)
    
    if len(visited) == N:
        return strongest
    else:
        return -1

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    relations = []
    for _ in range(M):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        relations.append((A, B))
    
    result = find_strongest_programmer(N, M, relations)
    print(result)