# YOUR CODE HERE
def main():
    import sys

    # Read input
    input = sys.stdin.read().splitlines()
    idx = 0

    # Read sheet A
    H_A, W_A = map(int, input[idx].split())
    idx +=1
    A = [input[idx + i] for i in range(H_A)]
    idx += H_A

    # Read sheet B
    H_B, W_B = map(int, input[idx].split())
    idx +=1
    B = [input[idx + i] for i in range(H_B)]
    idx += H_B

    # Read sheet X
    H_X, W_X = map(int, input[idx].split())
    idx +=1
    X = [input[idx + i] for i in range(H_X)]
    idx += H_X

    # Extract black squares
    A_black = []
    for r in range(H_A):
        for c in range(W_A):
            if A[r][c] == '#':
                A_black.append( (r, c) )

    B_black = []
    for r in range(H_B):
        for c in range(W_B):
            if B[r][c] == '#':
                B_black.append( (r, c) )

    X_black = set()
    for r in range(H_X):
        for c in range(W_X):
            if X[r][c] == '#':
                X_black.add( (r, c) )

    # Function to get all valid shifted positions for a sheet
    def get_shifted(sheet_black, H_sheet, W_sheet):
        shifted_list = []
        for dr in range(-H_X, H_X):
            for dc in range(-W_X, W_X):
                shifted = set()
                valid = True
                for (r, c) in sheet_black:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < H_X and 0 <= nc < W_X:
                        shifted.add( (nr, nc) )
                    else:
                        valid = False
                        break
                if valid:
                    shifted_list.append(shifted)
        return shifted_list

    # Get all possible shifted positions for A and B
    shifted_A_list = get_shifted(A_black, H_A, W_A)
    shifted_B_list = get_shifted(B_black, H_B, W_B)

    # Iterate over all combinations
    for shifted_A in shifted_A_list:
        for shifted_B in shifted_B_list:
            combined = shifted_A.union(shifted_B)
            if combined == X_black:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()