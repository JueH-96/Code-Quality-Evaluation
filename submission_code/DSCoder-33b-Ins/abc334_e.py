import sys

MOD = 998244353

def read_input():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]
    return H, W, S

def solve(H, W, S):
    # Calculate the number of green cells
    green_cells = sum(row.count('#') for row in S)

    # Calculate the number of red cells
    red_cells = sum(row.count('.') for row in S)

    # Calculate the expected value
    expected_value = green_cells * (green_cells - 1) // 2 + green_cells * red_cells

    # Return the expected value modulo MOD
    return expected_value % MOD

def main():
    H, W, S = read_input()
    print(solve(H, W, S))

if __name__ == '__main__':
    main()