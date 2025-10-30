from itertools import permutations

def min_swaps_to_permute(target):
    """Minimum adjacent swaps to transform [0,1,2,...,n-1] to target"""
    n = len(target)
    current = list(range(n))
    swaps = 0
    
    for i in range(n):
        while current[i] != target[i]:
            # Find where target[i] is in current
            j = i + 1
            while j < n and current[j] != target[i]:
                j += 1
            # Swap adjacent elements to bring target[i] to position i
            while j > i:
                current[j], current[j-1] = current[j-1], current[j]
                swaps += 1
                j -= 1
    
    return swaps

H, W = map(int, input().split())

A = []
for _ in range(H):
    row = list(map(int, input().split()))
    A.append(row)

B = []
for _ in range(H):
    row = list(map(int, input().split()))
    B.append(row)

min_ops = float('inf')
found = False

# Try all permutations of rows and columns
for row_perm in permutations(range(H)):
    for col_perm in permutations(range(W)):
        # Check if this permutation transforms A to B
        valid = True
        for i in range(H):
            for j in range(W):
                if A[row_perm[i]][col_perm[j]] != B[i][j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            found = True
            # Count swaps needed
            row_swaps = min_swaps_to_permute(list(row_perm))
            col_swaps = min_swaps_to_permute(list(col_perm))
            total_swaps = row_swaps + col_swaps
            min_ops = min(min_ops, total_swaps)

if found:
    print(min_ops)
else:
    print(-1)