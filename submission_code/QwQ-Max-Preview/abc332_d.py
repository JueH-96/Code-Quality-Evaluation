import sys
from itertools import permutations

def inversion_count(perm):
    count = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                count += 1
    return count

def main():
    H, W = map(int, sys.stdin.readline().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))
    B = []
    for _ in range(H):
        B.append(list(map(int, sys.stdin.readline().split())))
    
    # Check if the elements are the same
    a_flat = [num for row in A for num in row]
    b_flat = [num for row in B for num in row]
    if sorted(a_flat) != sorted(b_flat):
        print(-1)
        return
    
    min_swaps = None
    
    # Generate all possible row permutations and column permutations
    for row_perm in permutations(range(H)):
        for col_perm in permutations(range(W)):
            # Apply the permutations to grid A
            transformed = []
            for r in row_perm:
                original_row = A[r]
                transformed_row = [original_row[c] for c in col_perm]
                transformed.append(transformed_row)
            
            if transformed == B:
                # Calculate the inversion counts
                row_swaps = inversion_count(row_perm)
                col_swaps = inversion_count(col_perm)
                total = row_swaps + col_swaps
                if min_swaps is None or total < min_swaps:
                    min_swaps = total
    
    print(min_swaps if min_swaps is not None else -1)

if __name__ == "__main__":
    main()