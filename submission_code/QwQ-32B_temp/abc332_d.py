import sys
from itertools import permutations

def inversion_count(p):
    count = 0
    n = len(p)
    for i in range(n):
        for j in range(i+1, n):
            if p[i] > p[j]:
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
    
    min_ops = float('inf')
    
    for rows in permutations(range(H)):
        row_permuted = [A[i] for i in rows]
        for cols in permutations(range(W)):
            transformed = []
            for r in row_permuted:
                new_row = [r[j] for j in cols]
                transformed.append(new_row)
            if transformed == B:
                inv_rows = inversion_count(rows)
                inv_cols = inversion_count(cols)
                total = inv_rows + inv_cols
                if total < min_ops:
                    min_ops = total
    
    print(-1 if min_ops == float('inf') else min_ops)

if __name__ == "__main__":
    main()