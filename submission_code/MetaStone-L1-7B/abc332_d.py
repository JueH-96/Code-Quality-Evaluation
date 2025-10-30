import sys
from itertools import permutations

def count_inversions(perm):
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    return inv

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1

    A = []
    for _ in range(H):
        row = list(map(int, input[ptr:ptr+W]))
        ptr += W
        A.append(row)

    B = []
    for _ in range(H):
        row = list(map(int, input[ptr:ptr+W]))
        ptr += W
        B.append(row)

    min_ops = None

    for row_perm in permutations(range(H)):
        grid_row = [A[i] for i in row_perm]

        for col_perm in permutations(range(W)):
            grid_col = []
            for j in range(W):
                new_col = []
                for i in range(H):
                    new_col.append(grid_row[i][col_perm[j]])
                grid_col.append(new_col)

            if grid_col == B:
                row_swaps = count_inversions(row_perm)
                col_swaps = count_inversions(col_perm)
                total = row_swaps + col_swaps
                if min_ops is None or total < min_ops:
                    min_ops = total

    if min_ops is not None:
        print(min_ops)
    else:
        print(-1)

if __name__ == '__main__':
    main()