import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = data[3:]
    INF = 10**18
    ans = INF

    # Check horizontal segments if possible
    if W >= K:
        for i in range(H):
            row = grid[i]
            sum_x = 0
            sum_dot = 0
            # initial window [0..K-1]
            for j in range(K):
                c = row[j]
                if c == 'x':
                    sum_x += 1
                elif c == '.':
                    sum_dot += 1
            if sum_x == 0:
                ans = min(ans, sum_dot)
            # slide window
            for j in range(K, W):
                c_out = row[j-K]
                if c_out == 'x':
                    sum_x -= 1
                elif c_out == '.':
                    sum_dot -= 1
                c_in = row[j]
                if c_in == 'x':
                    sum_x += 1
                elif c_in == '.':
                    sum_dot += 1
                if sum_x == 0:
                    ans = min(ans, sum_dot)

    # Check vertical segments if possible
    if H >= K:
        for j in range(W):
            sum_x = 0
            sum_dot = 0
            # initial window [0..K-1] in column j
            for i in range(K):
                c = grid[i][j]
                if c == 'x':
                    sum_x += 1
                elif c == '.':
                    sum_dot += 1
            if sum_x == 0:
                ans = min(ans, sum_dot)
            # slide window down
            for i in range(K, H):
                c_out = grid[i-K][j]
                if c_out == 'x':
                    sum_x -= 1
                elif c_out == '.':
                    sum_dot -= 1
                c_in = grid[i][j]
                if c_in == 'x':
                    sum_x += 1
                elif c_in == '.':
                    sum_dot += 1
                if sum_x == 0:
                    ans = min(ans, sum_dot)

    if ans == INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()