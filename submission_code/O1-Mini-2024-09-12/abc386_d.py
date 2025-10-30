# YOUR CODE HERE
def main():
    import sys
    import sys
    import sys
    from collections import defaultdict
    input = sys.stdin.read
    data = input().split()
    N, M = int(data[0]), int(data[1])
    row_B = {}
    row_W = {}
    col_B = {}
    col_W = {}
    idx = 2
    for _ in range(M):
        X = int(data[idx])
        Y = int(data[idx+1])
        C = data[idx+2]
        idx +=3
        if C == 'B':
            if X in row_B:
                if Y > row_B[X]:
                    row_B[X] = Y
            else:
                row_B[X] = Y
            if Y in col_B:
                if X > col_B[Y]:
                    col_B[Y] = X
            else:
                col_B[Y] = X
        else:
            if X in row_W:
                if Y < row_W[X]:
                    row_W[X] = Y
            else:
                row_W[X] = Y
            if Y in col_W:
                if X < col_W[Y]:
                    col_W[Y] = X
            else:
                col_W[Y] = X
    # Check rows
    for r in row_B:
        if r in row_W:
            if row_B[r] >= row_W[r]:
                print("No")
                return
    for r in row_W:
        if r in row_B:
            if row_B[r] >= row_W[r]:
                print("No")
                return
    # Check columns
    for c in col_B:
        if c in col_W:
            if col_B[c] >= col_W[c]:
                print("No")
                return
    for c in col_W:
        if c in col_B:
            if col_B[c] >= col_W[c]:
                print("No")
                return
    print("Yes")

if __name__ == "__main__":
    main()