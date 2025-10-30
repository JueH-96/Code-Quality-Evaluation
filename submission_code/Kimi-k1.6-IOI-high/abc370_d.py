import sys
from collections import defaultdict

def main():
    Input = sys.stdin.read().split()
    ptr = 0
    H = int(Input[ptr])
    ptr += 1
    W = int(Input[ptr])
    ptr += 1
    Q = int(Input[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        R = int(Input[ptr])
        ptr += 1
        C = int(Input[ptr])
        ptr += 1
        queries.append((R, C))
    
    # Initialize wall status with a dummy boundary (rows 0 and H+1, cols 0 and W+1)
    # Rows and columns are 1-based
    wall_status = [[True] * (W + 2) for _ in range(H + 2)]  # wall_status[R][C] for R 0..H+1, C 0..W+1
    parent_row = defaultdict(dict)
    left_row = defaultdict(dict)
    right_row = defaultdict(dict)
    parent_col = defaultdict(dict)
    top_col = defaultdict(dict)
    bottom_col = defaultdict(dict)
    total_walls = H * W
    
    def find_row(R, C):
        if C not in parent_row[R]:
            return C
        path = []
        while parent_row[R].get(C, C) != C:
            path.append(C)
            C = parent_row[R][C]
        root = C
        for node in path:
            parent_row[R][node] = root
        return root
    
    def find_col(C, R):
        if R not in parent_col[C]:
            return R
        path = []
        while parent_col[C].get(R, R) != R:
            path.append(R)
            R = parent_col[C][R]
        root = R
        for node in path:
            parent_col[C][node] = root
        return root
    
    def union_row(R, C1, C2):
        root1 = find_row(R, C1)
        root2 = find_row(R, C2)
        if root1 == root2:
            return
        parent_row[R][root1] = root2
        left_row[R][root2] = min(left_row[R].get(root1, root1), left_row[R].get(root2, root2))
        right_row[R][root2] = max(right_row[R].get(root1, root1), right_row[R].get(root2, root2))
    
    def union_col(C, R1, R2):
        root1 = find_col(C, R1)
        root2 = find_col(C, R2)
        if root1 == root2:
            return
        parent_col[C][root1] = root2
        top_col[C][root2] = min(top_col[C].get(root1, root1), top_col[C].get(root2, root2))
        bottom_col[C][root2] = max(bottom_col[C].get(root1, root1), bottom_col[C].get(root2, root2))
    
    def destroy(R, C):
        nonlocal total_walls
        if R < 1 or R > H or C < 1 or C > W:
            return
        if wall_status[R][C]:
            total_walls -= 1
            wall_status[R][C] = False
            # Add to row R's union-find
            if C not in parent_row[R]:
                parent_row[R][C] = C
                left_row[R][C] = C
                right_row[R][C] = C
            # Union with left and right neighbors
            for nc in [C-1, C+1]:
                if 1 <= nc <= W and nc in parent_row[R]:
                    union_row(R, C, nc)
            # Add to column C's union-find
            if R not in parent_col[C]:
                parent_col[C][R] = R
                top_col[C][R] = R
                bottom_col[C][R] = R
            # Union with top and bottom neighbors
            for nr in [R-1, R+1]:
                if 1 <= nr <= H and nr in parent_col[C]:
                    union_col(C, R, nr)
    
    for R, C in queries:
        if wall_status[R][C]:
            destroy(R, C)
        else:
            # Check left direction in row R
            root_row_rc = find_row(R, C)
            L = left_row[R].get(root_row_rc, root_row_rc)
            R_bound = right_row[R].get(root_row_rc, root_row_rc)
            # Left candidate: L-1
            if L - 1 >= 1 and wall_status[R][L-1]:
                destroy(R, L-1)
            # Right candidate: R_bound + 1
            if R_bound + 1 <= W and wall_status[R][R_bound + 1]:
                destroy(R, R_bound + 1)
            
            # Check up and down directions in column C
            root_col_rc = find_col(C, R)
            A = top_col[C].get(root_col_rc, root_col_rc)
            B = bottom_col[C].get(root_col_rc, root_col_rc)
            # Up candidate: A-1
            if A - 1 >= 1 and wall_status[A-1][C]:
                destroy(A-1, C)
            # Down candidate: B+1
            if B + 1 <= H and wall_status[B+1][C]:
                destroy(B+1, C)
    
    print(total_walls)

if __name__ == "__main__":
    main()