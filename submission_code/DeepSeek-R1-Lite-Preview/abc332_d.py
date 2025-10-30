import itertools

def count_inversions(perm):
    inversions = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inversions += 1
    return inversions

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    index = 2
    A = []
    for _ in range(H):
        A.append(list(map(int, data[index:index+W])))
        index += W
    B = []
    for _ in range(H):
        B.append(list(map(int, data[index:index+W])))
        index += W
    
    min_operations = float('inf')
    
    row_indices = list(range(H))
    for row_perm in itertools.permutations(row_indices):
        row_permuted_A = [A[row_perm[i]] for i in range(H)]
        col_indices = list(range(W))
        for col_perm in itertools.permutations(col_indices):
            final_A = [[row_permuted_A[i][col_perm[j]] for j in range(W)] for i in range(H)]
            if final_A == B:
                row_swaps = count_inversions(row_perm)
                col_swaps = count_inversions(col_perm)
                total_operations = row_swaps + col_swaps
                if total_operations < min_operations:
                    min_operations = total_operations
    
    if min_operations != float('inf'):
        print(min_operations)
    else:
        print(-1)

if __name__ == "__main__":
    main()