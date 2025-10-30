import sys
from itertools import product

def backtrack(row, a, b, c, N, R, C):
    if row == N:
        if check_columns(a, b, c, C):
            grid = []
            for i in range(N):
                line = ['.'] * N
                line[a[i]] = 'A'
                line[b[i]] = 'B'
                line[c[i]] = 'C'
                grid.append(''.join(line))
            return grid
        else:
            return None
    available_a = [col for col in range(N) if col not in a]
    available_b = [col for col in range(N) if col not in b]
    available_c = [col for col in range(N) if col not in c]
    required_char = R[row]
    for col_a, col_b, col_c in product(available_a, available_b, available_c):
        if col_a == col_b or col_a == col_c or col_b == col_c:
            continue
        min_col = min(col_a, col_b, col_c)
        if (required_char == 'A' and min_col == col_a) or \
           (required_char == 'B' and min_col == col_b) or \
           (required_char == 'C' and min_col == col_c):
            new_a = a + [col_a]
            new_b = b + [col_b]
            new_c = c + [col_c]
            result = backtrack(row + 1, new_a, new_b, new_c, N, R, C)
            if result is not None:
                return result
    return None

def check_columns(a, b, c, C):
    N = len(a)
    for j in range(N):
        min_i = None
        char = None
        for i in range(N):
            if a[i] == j:
                if min_i is None or i < min_i:
                    min_i = i
                    char = 'A'
            if b[i] == j:
                if min_i is None or i < min_i:
                    min_i = i
                    char = 'B'
            if c[i] == j:
                if min_i is None or i < min_i:
                    min_i = i
                    char = 'C'
        if char != C[j]:
            return False
    return True

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    R = input[1]
    C = input[2]
    result = backtrack(0, [], [], [], N, R, C)
    if result:
        print("Yes")
        for row in result:
            print(row)
    else:
        print("No")

if __name__ == '__main__':
    main()