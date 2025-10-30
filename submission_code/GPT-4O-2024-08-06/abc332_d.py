from itertools import permutations

def count_swaps_to_sort(perm):
    visited = [False] * len(perm)
    swaps = 0
    for i in range(len(perm)):
        if visited[i] or perm[i] == i:
            continue
        cycle_size = 0
        x = i
        while not visited[x]:
            visited[x] = True
            x = perm[x]
            cycle_size += 1
        if cycle_size > 0:
            swaps += cycle_size - 1
    return swaps

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    
    A = []
    B = []
    
    index = 2
    for _ in range(H):
        A.append(list(map(int, data[index:index+W])))
        index += W
    
    for _ in range(H):
        B.append(list(map(int, data[index:index+W])))
        index += W
    
    min_operations = float('inf')
    
    row_permutations = list(permutations(range(H)))
    col_permutations = list(permutations(range(W)))
    
    for row_perm in row_permutations:
        for col_perm in col_permutations:
            # Create the permuted version of A
            permuted_A = [[0] * W for _ in range(H)]
            for i in range(H):
                for j in range(W):
                    permuted_A[i][j] = A[row_perm[i]][col_perm[j]]
            
            # Check if permuted_A matches B
            if permuted_A == B:
                row_swaps = count_swaps_to_sort(row_perm)
                col_swaps = count_swaps_to_sort(col_perm)
                total_swaps = row_swaps + col_swaps
                min_operations = min(min_operations, total_swaps)
    
    if min_operations == float('inf'):
        print(-1)
    else:
        print(min_operations)