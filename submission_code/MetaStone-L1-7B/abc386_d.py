import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    cells = []
    rows = defaultdict(list)
    cols = defaultdict(list)
    
    for _ in range(M):
        x, y, c = sys.stdin.readline().split()
        x = int(x)
        y = int(y)
        cells.append((x, y, c))
        rows[x].append(y)
        cols[y].append(x)
    
    # Compute i_x for each row
    i_x_dict = {}
    for x in rows:
        B_ys = []
        W_ys = []
        for y in rows[x]:
            if cells[rows[x].index(y)][2] == 'B':
                B_ys.append(y)
            else:
                W_ys.append(y)
        if B_ys:
            y_b = max(B_ys)
        else:
            y_b = 0
        if W_ys:
            y_w = min(W_ys)
            if y_b >= y_w:
                print("No")
                return
            i_x = y_b
        else:
            i_x = N
        i_x_dict[x] = i_x
    
    # Compute j_y for each column
    j_y_dict = {}
    for y in cols:
        B_xs = []
        W_xs = []
        for x in cols[y]:
            idx = cells.index((x, y, cells[0]))
            c = cells[idx][2]
            if c == 'B':
                B_xs.append(x)
            else:
                W_xs.append(x)
        if B_xs:
            x_b = max(B_xs)
        else:
            x_b = 0
        if W_xs:
            x_w = min(W_xs)
            if x_b >= x_w:
                print("No")
                return
            j_y = x_b
        else:
            j_y = N
        j_y_dict[y] = j_y
    
    # Check if j_max >= i_max
    i_max = max(i_x_dict.values()) if i_x_dict else 0
    j_max = max(j_y_dict.values()) if j_y_dict else 0
    if j_max < i_max:
        print("No")
        return
    
    # Check all given cells
    for x, y, c in cells:
        i_x = i_x_dict[x]
        j_y = j_y_dict[y]
        if c == 'B':
            if y > i_x or x > j_y:
                print("No")
                return
        else:
            if y <= i_x or x <= j_y:
                print("No")
                return
    
    print("Yes")

if __name__ == '__main__':
    main()