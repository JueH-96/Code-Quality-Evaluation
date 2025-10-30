import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr]); ptr += 1
    W = int(input[ptr]); ptr += 1
    N = int(input[ptr]); ptr += 1

    # Initialize hole grid
    hole_grid = [[False] * (W + 2) for _ in range(H + 2)]
    for _ in range(N):
        a = int(input[ptr]); ptr += 1
        b = int(input[ptr]); ptr += 1
        hole_grid[a][b] = True

    # DP arrays
    prev_row = [0] * (W + 2)
    total = 0

    for i in range(1, H + 1):
        curr_row = [0] * (W + 2)
        hole_row = hole_grid[i]
        pr = prev_row
        cr = curr_row
        for j in range(1, W + 1):
            if hole_row[j]:
                cr[j] = 0
            else:
                cr[j] = min(pr[j], cr[j - 1], pr[j - 1]) + 1
                total += cr[j]
        prev_row = curr_row[:]

    print(total)

if __name__ == "__main__":
    main()