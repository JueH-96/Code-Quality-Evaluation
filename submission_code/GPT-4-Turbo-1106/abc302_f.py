from collections import defaultdict

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def min_operations_to_get_set(N, M, sets):
    parent = list(range(M + 1))
    rank = [0] * (M + 1)
    element_to_set = defaultdict(list)
    
    for i, s in enumerate(sets):
        for element in s:
            element_to_set[element].append(i)
    
    for element, set_indices in element_to_set.items():
        for i in range(1, len(set_indices)):
            union(parent, rank, set_indices[0], set_indices[i])
    
    set_1 = find(parent, 1)
    set_M = find(parent, M)
    
    if set_1 != set_M:
        return -1
    
    return sum(1 for s in sets if find(parent, s[0]) == set_1) - 1

def main():
    N, M = map(int, input().split())
    sets = []
    for _ in range(N):
        A_i = int(input())
        S_i = list(map(int, input().split()))
        sets.append(S_i)
    
    result = min_operations_to_get_set(N, M, sets)
    print(result)

main()