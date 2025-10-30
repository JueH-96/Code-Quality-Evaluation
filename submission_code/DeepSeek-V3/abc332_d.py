from itertools import permutations

def main():
    H, W = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    B = [list(map(int, input().split())) for _ in range(H)]
    
    # Check if the multisets of elements are the same
    from collections import defaultdict
    count_A = defaultdict(int)
    count_B = defaultdict(int)
    for row in A:
        for num in row:
            count_A[num] += 1
    for row in B:
        for num in row:
            count_B[num] += 1
    if count_A != count_B:
        print(-1)
        return
    
    # Generate all possible row permutations
    row_perms = list(permutations(range(H)))
    # Generate all possible column permutations
    col_perms = list(permutations(range(W)))
    
    min_ops = float('inf')
    
    for rp in row_perms:
        for cp in col_perms:
            # Create a new grid based on the current row and column permutations
            new_A = [[0] * W for _ in range(H)]
            for i in range(H):
                for j in range(W):
                    new_A[i][j] = A[rp[i]][cp[j]]
            # Check if the new grid matches B
            if new_A == B:
                # Calculate the number of swaps for rows
                row_swaps = 0
                current_rp = list(rp)
                target_rp = list(range(H))
                for i in range(H):
                    if current_rp[i] != target_rp[i]:
                        idx = current_rp.index(target_rp[i])
                        current_rp[i], current_rp[idx] = current_rp[idx], current_rp[i]
                        row_swaps += 1
                # Calculate the number of swaps for columns
                col_swaps = 0
                current_cp = list(cp)
                target_cp = list(range(W))
                for j in range(W):
                    if current_cp[j] != target_cp[j]:
                        idx = current_cp.index(target_cp[j])
                        current_cp[j], current_cp[idx] = current_cp[idx], current_cp[j]
                        col_swaps += 1
                total_ops = row_swaps + col_swaps
                if total_ops < min_ops:
                    min_ops = total_ops
    
    if min_ops != float('inf'):
        print(min_ops)
    else:
        print(-1)

if __name__ == "__main__":
    main()