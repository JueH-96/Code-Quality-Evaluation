import itertools

def main():
    h, w = map(int, input().split())
    a = [list(map(int, input().split())) for _ in range(h)]
    b = [list(map(int, input().split())) for _ in range(h)]
    
    # Check if multisets are the same
    flat_a = [num for row in a for num in row]
    flat_b = [num for row in b for num in row]
    if sorted(flat_a) != sorted(flat_b):
        print(-1)
        return
    
    min_ops = float('inf')
    rows = list(range(h))
    cols = list(range(w))
    
    # Precompute all row and column permutations
    row_perms = itertools.permutations(rows)
    col_perms = itertools.permutations(cols)
    
    for rp in row_perms:
        for cp in col_perms:
            valid = True
            for i in range(h):
                for j in range(w):
                    if a[rp[i]][cp[j]] != b[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # Calculate inversion counts
                inv_row = 0
                for i in range(h):
                    for j in range(i+1, h):
                        if rp[i] > rp[j]:
                            inv_row += 1
                inv_col = 0
                for i in range(w):
                    for j in range(i+1, w):
                        if cp[i] > cp[j]:
                            inv_col += 1
                total = inv_row + inv_col
                if total < min_ops:
                    min_ops = total
    
    print(min_ops if min_ops != float('inf') else -1)

if __name__ == "__main__":
    main()