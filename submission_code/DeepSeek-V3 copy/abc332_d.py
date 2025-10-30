from itertools import permutations

def count_swaps(arr, target):
    arr = list(arr)
    swaps = 0
    for i in range(len(arr)):
        if arr[i] != target[i]:
            idx = arr.index(target[i], i)
            arr[i], arr[idx] = arr[idx], arr[i]
            swaps += 1
    return swaps

def main():
    H, W = map(int, input().split())
    A = [tuple(map(int, input().split())) for _ in range(H)]
    B = [tuple(map(int, input().split())) for _ in range(H)]
    
    # Check if the multisets of rows and columns are the same
    A_rows = [tuple(row) for row in A]
    B_rows = [tuple(row) for row in B]
    if sorted(A_rows) != sorted(B_rows):
        print(-1)
        return
    
    # Check if the multisets of columns are the same
    A_cols = [tuple(A[i][j] for i in range(H)) for j in range(W)]
    B_cols = [tuple(B[i][j] for i in range(H)) for j in range(W)]
    if sorted(A_cols) != sorted(B_cols):
        print(-1)
        return
    
    # Find the row permutation
    row_order = []
    for b_row in B_rows:
        idx = A_rows.index(b_row)
        row_order.append(idx)
    
    # Find the column permutation
    col_order = []
    for b_col in B_cols:
        idx = A_cols.index(b_col)
        col_order.append(idx)
    
    # Calculate the number of swaps for rows
    row_swaps = count_swaps(range(H), row_order)
    
    # Calculate the number of swaps for columns
    col_swaps = count_swaps(range(W), col_order)
    
    print(row_swaps + col_swaps)

if __name__ == "__main__":
    main()