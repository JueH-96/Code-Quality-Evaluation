def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    S = []
    for _ in range(H):
        S.append(input[idx])
        idx += 1
    
    min_ops = float('inf')
    
    # Process rows
    for row in S:
        if W < K:
            continue
        # Compute prefix_x and prefix_dot for this row
        prefix_x = [0] * (W + 1)
        prefix_dot = [0] * (W + 1)
        for i in range(W):
            prefix_x[i+1] = prefix_x[i] + (1 if row[i] == 'x' else 0)
            prefix_dot[i+1] = prefix_dot[i] + (1 if row[i] == '.' else 0)
        max_j = W - K
        for j in range(max_j + 1):
            x_count = prefix_x[j + K] - prefix_x[j]
            if x_count == 0:
                dot_count = prefix_dot[j + K] - prefix_dot[j]
                if dot_count < min_ops:
                    min_ops = dot_count
    
    # Process columns
    for c in range(W):
        if H < K:
            continue
        # Extract the column and compute prefix_x and prefix_dot
        prefix_x = [0] * (H + 1)
        prefix_dot = [0] * (H + 1)
        for i in range(H):
            ch = S[i][c]
            prefix_x[i+1] = prefix_x[i] + (1 if ch == 'x' else 0)
            prefix_dot[i+1] = prefix_dot[i] + (1 if ch == '.' else 0)
        max_i = H - K
        for i in range(max_i + 1):
            x_count = prefix_x[i + K] - prefix_x[i]
            if x_count == 0:
                dot_count = prefix_dot[i + K] - prefix_dot[i]
                if dot_count < min_ops:
                    min_ops = dot_count
    
    if min_ops == float('inf'):
        print(-1)
    else:
        print(min_ops)

if __name__ == '__main__':
    main()