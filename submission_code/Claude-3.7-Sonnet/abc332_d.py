from itertools import permutations

def count_inversions(p):
    """Count the number of inversions in a permutation"""
    inversions = 0
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            if p[i] > p[j]:
                inversions += 1
    return inversions

def main():
    # Read input
    H, W = map(int, input().split())
    
    # Read grid A
    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))
    
    # Read grid B
    B = []
    for _ in range(H):
        B.append(list(map(int, input().split())))
    
    # Try all possible permutations of rows and columns
    min_operations = float('inf')
    for row_perm in permutations(range(H)):
        for col_perm in permutations(range(W)):
            # Check if this permutation transforms A into B
            matches = True
            for i in range(H):
                for j in range(W):
                    if A[row_perm[i]][col_perm[j]] != B[i][j]:
                        matches = False
                        break
                if not matches:
                    break
            
            # If it matches, calculate the number of operations needed
            if matches:
                row_inversions = count_inversions(row_perm)
                col_inversions = count_inversions(col_perm)
                total_operations = row_inversions + col_inversions
                min_operations = min(min_operations, total_operations)
    
    # Output the result
    if min_operations == float('inf'):
        print(-1)  # Not possible to transform A into B
    else:
        print(min_operations)

if __name__ == "__main__":
    main()