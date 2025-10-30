import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, N = map(int, sys.stdin.readline().split())
    bars = []
    for _ in range(N):
        R, C, L = map(int, sys.stdin.readline().split())
        bars.append((R, C, L))
    
    # Initialize initial_row for each column (1-based)
    initial_row = [0] * (W + 2)  # columns 1..W
    for R, C, L in bars:
        for j in range(C, C + L):
            initial_row[j] = R  # overwrite; input ensures no overlaps

    # down[j] is the first row below initial bars + 1
    down = [H + 1] * (W + 2)  # 1-based indexing
    for j in range(1, W + 1):
        if initial_row[j] != 0:
            down[j] = initial_row[j] + 1
        else:
            down[j] = H + 1  # no initial bar in this column

    result = [0] * N

    for idx in range(N):
        R_i, C_i, L_i = bars[idx]
        min_down = H + 2
        # Find the minimum down[j] in the span
        for j in range(C_i, C_i + L_i):
            if down[j] < min_down:
                min_down = down[j]
        new_r = min_down - 1
        if new_r < R_i:
            new_r = R_i
        result[idx] = new_r
        # Update down[j] to new_r + 1
        val = new_r + 1
        for j in range(C_i, C_i + L_i):
            down[j] = val

    for r in result:
        print(r)

threading.Thread(target=main).start()