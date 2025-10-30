def main():
    # Parse input
    H_A, W_A = map(int, input().split())
    A = [input() for _ in range(H_A)]
    
    H_B, W_B = map(int, input().split())
    B = [input() for _ in range(H_B)]
    
    H_X, W_X = map(int, input().split())
    X = [input() for _ in range(H_X)]
    
    # Check if the goal is achievable
    if is_goal_achievable(A, B, X):
        print("Yes")
    else:
        print("No")

def is_goal_achievable(A, B, X):
    H_A, W_A = len(A), len(A[0])
    H_B, W_B = len(B), len(B[0])
    H_X, W_X = len(X), len(X[0])
    
    # Try all possible relative positions of A and B with respect to the cut-out region
    for A_top in range(-H_A + 1, H_X):
        for A_left in range(-W_A + 1, W_X):
            for B_top in range(-H_B + 1, H_X):
                for B_left in range(-W_B + 1, W_X):
                    # Check if all black squares of A and B are within the bounds of the cut-out region
                    if not all_black_squares_inside(A, A_top, A_left, H_X, W_X) or not all_black_squares_inside(B, B_top, B_left, H_X, W_X):
                        continue
                    
                    # Create the cut-out region
                    cut_out = create_cut_out(A, A_top, A_left, B, B_top, B_left, H_X, W_X)
                    
                    # Check if the cut-out region matches X
                    if cut_out_matches(cut_out, X):
                        return True
    
    return False

def all_black_squares_inside(sheet, top, left, H, W):
    for i in range(len(sheet)):
        for j in range(len(sheet[0])):
            if sheet[i][j] == '#' and not (0 <= top + i < H and 0 <= left + j < W):
                return False
    return True

def create_cut_out(A, A_top, A_left, B, B_top, B_left, H, W):
    cut_out = [['.' for _ in range(W)] for _ in range(H)]
    
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == '#' and 0 <= A_top + i < H and 0 <= A_left + j < W:
                cut_out[A_top + i][A_left + j] = '#'
    
    for i in range(len(B)):
        for j in range(len(B[0])):
            if B[i][j] == '#' and 0 <= B_top + i < H and 0 <= B_left + j < W:
                cut_out[B_top + i][B_left + j] = '#'
    
    return cut_out

def cut_out_matches(cut_out, X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            if cut_out[i][j] != X[i][j]:
                return False
    return True

if __name__ == "__main__":
    main()