# YOUR CODE HERE
def main():
    H_A, W_A = map(int, input().split())
    A = [input() for _ in range(H_A)]
    H_B, W_B = map(int, input().split())
    B = [input() for _ in range(H_B)]
    H_X, W_X = map(int, input().split())
    X = [input() for _ in range(H_X)]

    H_MAX = 10  # Maximum height and width as per constraints
    W_MAX = 10

    # sya and sxa are the shifts for A
    # syb and sxb are the shifts for B
    for sya in range(-H_A +1, H_X):
        for sxa in range(-W_A +1, W_X):
            for syb in range(-H_B +1, H_X):
                for sxb in range(-W_B +1, W_X):
                    D = [['.' for _ in range(W_X)] for _ in range(H_X)]
                    # Place A onto D
                    for i in range(H_A):
                        for j in range(W_A):
                            if A[i][j] == '#':
                                y = i + sya
                                x = j + sxa
                                if 0 <= y < H_X and 0 <= x < W_X:
                                    D[y][x] = '#'
                    # Place B onto D
                    for i in range(H_B):
                        for j in range(W_B):
                            if B[i][j] == '#':
                                y = i + syb
                                x = j + sxb
                                if 0 <= y < H_X and 0 <= x < W_X:
                                    D[y][x] = '#'
                    # Compare D and X
                    match = True
                    for i in range(H_X):
                        for j in range(W_X):
                            if D[i][j] != X[i][j]:
                                match = False
                                break
                        if not match:
                            break
                    if match:
                        print("Yes")
                        return  # Exit on first success
    print("No")

if __name__ == "__main__":
    main()