# YOUR CODE HERE
import sys
from itertools import permutations

def read_input():
    input = sys.stdin.read().strip().split()
    H = int(input[0])
    W = int(input[1])
    A = []
    B = []
    index = 2
    for i in range(H):
        A.append(list(map(int, input[index:index + W])))
        index += W
    for i in range(H):
        B.append(list(map(int, input[index:index + W])))
        index += W
    return H, W, A, B

def is_permutation_possible(A_row, B_row):
    return sorted(A_row) == sorted(B_row)

def min_swaps_to_match(A_row, B_row):
    if A_row == B_row:
        return 0
    min_swaps = float('inf')
    for perm in permutations(A_row):
        if list(perm) == B_row:
            swaps = 0
            temp = list(A_row)
            for i in range(len(A_row)):
                while temp[i] != B_row[i]:
                    swap_index = temp.index(B_row[i])
                    temp[i], temp[swap_index] = temp[swap_index], temp[i]
                    swaps += 1
            min_swaps = min(min_swaps, swaps)
    return min_swaps if min_swaps != float('inf') else -1

def min_operations_to_match(A, B, H, W):
    total_operations = 0
    row_permutations = []
    for i in range(H):
        if not is_permutation_possible(A[i], B[i]):
            return -1
        row_permutations.append(min_swaps_to_match(A[i], B[i]))
    
    col_permutations = []
    for j in range(W):
        A_col = [A[i][j] for i in range(H)]
        B_col = [B[i][j] for i in range(H)]
        if not is_permutation_possible(A_col, B_col):
            return -1
        col_permutations.append(min_swaps_to_match(A_col, B_col))
    
    return sum(row_permutations) + sum(col_permutations)

def main():
    H, W, A, B = read_input()
    result = min_operations_to_match(A, B, H, W)
    print(result)

if __name__ == "__main__":
    main()