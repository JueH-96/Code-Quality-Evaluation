def main():
    import sys
    input = sys.stdin.readline

    H, W, K = map(int, input().split())
    S = [input().rstrip() for _ in range(H)]

    # Initialize answer to a large value
    INF = H * W + 5
    ans = INF

    # Check all horizontal segments of length K
    if W >= K:
        for i in range(H):
            row = S[i]
            x_count = 0
            dot_count = 0
            # Initial window [0..K-1]
            for j in range(K):
                c = row[j]
                if c == 'x':
                    x_count += 1
                elif c == '.':
                    dot_count += 1
            if x_count == 0:
                ans = min(ans, dot_count)
            # Slide the window
            for j in range(K, W):
                # remove j-K, add j
                old = row[j-K]
                new = row[j]
                if old == 'x':
                    x_count -= 1
                elif old == '.':
                    dot_count -= 1
                if new == 'x':
                    x_count += 1
                elif new == '.':
                    dot_count += 1
                if x_count == 0:
                    ans = min(ans, dot_count)

    # Check all vertical segments of length K
    if H >= K:
        for j in range(W):
            x_count = 0
            dot_count = 0
            # Initial window [0..K-1] in column j
            for i in range(K):
                c = S[i][j]
                if c == 'x':
                    x_count += 1
                elif c == '.':
                    dot_count += 1
            if x_count == 0:
                ans = min(ans, dot_count)
            # Slide the window down the column
            for i in range(K, H):
                old = S[i-K][j]
                new = S[i][j]
                if old == 'x':
                    x_count -= 1
                elif old == '.':
                    dot_count -= 1
                if new == 'x':
                    x_count += 1
                elif new == '.':
                    dot_count += 1
                if x_count == 0:
                    ans = min(ans, dot_count)

    # Output result
    if ans == INF:
        print(-1)
    else:
        print(ans)

# Call main to execute
main()