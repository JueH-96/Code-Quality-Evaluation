import sys

def main():
    # Read input for A
    HA, WA = map(int, sys.stdin.readline().split())
    A_black = []
    for i in range(HA):
        line = sys.stdin.readline().strip()
        for j in range(WA):
            if line[j] == '#':
                A_black.append((i, j))
    
    # Read input for B
    HB, WB = map(int, sys.stdin.readline().split())
    B_black = []
    for i in range(HB):
        line = sys.stdin.readline().strip()
        for j in range(WB):
            if line[j] == '#':
                B_black.append((i, j))
    
    # Read input for X
    HX, WX = map(int, sys.stdin.readline().split())
    X_black = set()
    for i in range(HX):
        line = sys.stdin.readline().strip()
        for j in range(WX):
            if line[j] == '#':
                X_black.add((i, j))
    
    # Check if A and B have at least one black cell (per problem constraints)
    if not A_black or not B_black:
        print("No")
        return
    
    # Calculate valid ranges for A's offset
    min_dx_a = max(-i for i, j in A_black)
    max_dx_a = min((HX - 1 - i) for i, j in A_black)
    min_dy_a = max(-j for i, j in A_black)
    max_dy_a = min((WX - 1 - j) for i, j in A_black)
    valid_a = (min_dx_a <= max_dx_a) and (min_dy_a <= max_dy_a)
    
    # Calculate valid ranges for B's offset
    min_dx_b = max(-i for i, j in B_black)
    max_dx_b = min((HX - 1 - i) for i, j in B_black)
    min_dy_b = max(-j for i, j in B_black)
    max_dy_b = min((WX - 1 - j) for i, j in B_black)
    valid_b = (min_dx_b <= max_dx_b) and (min_dy_b <= max_dy_b)
    
    if not (valid_a and valid_b):
        print("No")
        return
    
    # Iterate all valid positions for A and B
    found = False
    for dx_a in range(min_dx_a, max_dx_a + 1):
        for dy_a in range(min_dy_a, max_dy_a + 1):
            a_cells = set()
            for (i, j) in A_black:
                x = dx_a + i
                y = dy_a + j
                a_cells.add((x, y))
            for dx_b in range(min_dx_b, max_dx_b + 1):
                for dy_b in range(min_dy_b, max_dy_b + 1):
                    b_cells = set()
                    for (i, j) in B_black:
                        x = dx_b + i
                        y = dy_b + j
                        b_cells.add((x, y))
                    union = a_cells.union(b_cells)
                    if union == X_black:
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()