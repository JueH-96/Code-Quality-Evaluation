import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    H, W, K = map(int, input().split())
    S = [input().rstrip('
') for _ in range(H)]

    INF = 10**18
    ans = INF

    # Check rows
    if K <= W:
        for i in range(H):
            cnt_x = 0
            cnt_dot = 0
            # initialize first window
            for j in range(K):
                c = S[i][j]
                if c == 'x':
                    cnt_x += 1
                elif c == '.':
                    cnt_dot += 1
            if cnt_x == 0:
                ans = min(ans, cnt_dot)
            # slide window
            for j in range(K, W):
                # add S[i][j]
                c = S[i][j]
                if c == 'x':
                    cnt_x += 1
                elif c == '.':
                    cnt_dot += 1
                # remove S[i][j-K]
                c_old = S[i][j-K]
                if c_old == 'x':
                    cnt_x -= 1
                elif c_old == '.':
                    cnt_dot -= 1
                if cnt_x == 0:
                    ans = min(ans, cnt_dot)

    # Check columns
    if K <= H:
        for j in range(W):
            cnt_x = 0
            cnt_dot = 0
            # initialize first window
            for i in range(K):
                c = S[i][j]
                if c == 'x':
                    cnt_x += 1
                elif c == '.':
                    cnt_dot += 1
            if cnt_x == 0:
                ans = min(ans, cnt_dot)
            # slide window
            for i in range(K, H):
                # add S[i][j]
                c = S[i][j]
                if c == 'x':
                    cnt_x += 1
                elif c == '.':
                    cnt_dot += 1
                # remove S[i-K][j]
                c_old = S[i-K][j]
                if c_old == 'x':
                    cnt_x -= 1
                elif c_old == '.':
                    cnt_dot -= 1
                if cnt_x == 0:
                    ans = min(ans, cnt_dot)

    if ans == INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()