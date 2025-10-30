import sys
import itertools

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    H, W = read_ints()
    A = [read_ints() for _ in range(H)]
    B = [read_ints() for _ in range(H)]
    
    # To solve the problem, we need to check if by any permutation of rows and columns
    # we can transform A into B. Since H, W <= 5, we can afford to try all permutations.
    
    # Generate all row permutations and column permutations
    row_perms = list(itertools.permutations(range(H)))
    col_perms = list(itertools.permutations(range(W)))
    
    # Function to apply a row permutation and column permutation to A
    def apply_permutations(row_perm, col_perm):
        # Apply row permutation
        new_grid = [A[row_perm[i]] for i in range(H)]
        # Apply column permutation
        new_grid = [[new_grid[i][col_perm[j]] for j in range(W)] for i in range(H)]
        return new_grid
    
    # Check all combinations of row and column permutations
    for row_perm in row_perms:
        for col_perm in col_perms:
            if apply_permutations(row_perm, col_perm) == B:
                # Calculate the minimum number of swaps to achieve these permutations
                # Count the number of swaps needed to get each permutation from the identity
                # Using bubble sort logic to count inversions
                def count_swaps(perm):
                    perm = list(perm)
                    count = 0
                    for i in range(len(perm)):
                        for j in range(i + 1, len(perm)):
                            if perm[i] > perm[j]:
                                perm[i], perm[j] = perm[j], perm[i]
                                count += 1
                    return count
                
                row_swaps = count_swaps(row_perm)
                col_swaps = count_swaps(col_perm)
                print(row_swaps + col_swaps)
                return
    
    # If no permutation matches, it's impossible
    print(-1)

if __name__ == "__main__":
    main()