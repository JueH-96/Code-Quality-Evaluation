import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    rows = defaultdict(lambda: {'max_B': -float('inf'), 'min_W_minus': float('inf'), 'has_B': False, 'has_W': False})
    cols = defaultdict(lambda: {'max_B': -float('inf'), 'min_W_minus': float('inf'), 'has_B': False, 'has_W': False})

    for _ in range(M):
        X, Y, C = sys.stdin.readline().split()
        X = int(X)
        Y = int(Y)
        if C == 'B':
            # Update row's B
            r = rows[X]
            if Y > r['max_B']:
                r['max_B'] = Y
            r['has_B'] = True
            # Update column's B
            c = cols[Y]
            if X > c['max_B']:
                c['max_B'] = X
            c['has_B'] = True
        else:
            # Update row's W
            r = rows[X]
            temp = Y - 1
            if temp < r['min_W_minus']:
                r['min_W_minus'] = temp
            r['has_W'] = True
            # Update column's W
            c = cols[Y]
            temp_col = X - 1
            if temp_col < c['min_W_minus']:
                c['min_W_minus'] = temp_col
            c['has_W'] = True

    # Check rows
    for x in rows:
        r = rows[x]
        has_B = r['has_B']
        has_W = r['has_W']
        max_B = r['max_B']
        min_W_minus = r['min_W_minus']
        if has_B:
            i_low = max_B
        else:
            i_low = 0
        if has_W:
            i_high = min_W_minus
        else:
            i_high = N
        if i_low > i_high:
            print("No")
            return

    # Check columns
    for y in cols:
        c = cols[y]
        has_B = c['has_B']
        has_W = c['has_W']
        max_B = c['max_B']
        min_W_minus = c['min_W_minus']
        if has_B:
            j_low = max_B
        else:
            j_low = 0
        if has_W:
            j_high = min_W_minus
        else:
            j_high = N
        if j_low > j_high:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    main()