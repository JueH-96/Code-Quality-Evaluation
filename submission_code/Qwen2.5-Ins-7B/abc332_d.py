# YOUR CODE HERE
H, W = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]
B = [list(map(int, input().split())) for _ in range(H)]

def count_swaps(A, B):
    swaps = 0
    for i in range(H):
        for j in range(W):
            if A[i][j] != B[i][j]:
                return float('inf')
    return swaps

def solve(A, B):
    min_swaps = float('inf')
    for i in range(H):
        for j in range(W):
            if A[i][j] != B[i][j]:
                for k in range(i+1, H):
                    A[i], A[k] = A[k], A[i]
                    swaps = count_swaps(A, B)
                    min_swaps = min(min_swaps, swaps)
                    A[i], A[k] = A[k], A[i]
                for k in range(j+1, W):
                    for l in range(i+1):
                        A[l][j], A[l][k] = A[l][k], A[l][j]
                        swaps = count_swaps(A, B)
                        min_swaps = min(min_swaps, swaps)
                        A[l][j], A[l][k] = A[l][j], A[l][k]
    return min_swaps if min_swaps != float('inf') else -1

print(solve(A, B))