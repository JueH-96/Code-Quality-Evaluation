def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    X = list(map(int, input[idx:idx+M]))
    idx += M
    
    S = 0
    diff = [0] * (N + 2)  # diff[0..N+1], but N+2 to be safe
    
    for i in range(M-1):
        A = X[i]
        B = X[i+1]
        a = A - 1  # 0-based
        b = B - 1  # 0-based
        
        cw = (b - a) % N
        ccw = (a - b) % N
        d = min(cw, ccw)
        S += d
        
        if cw < ccw:
            # Clockwise path: [a, (b-1) % N]
            L = a
            R = (b - 1) % N
            val = N - 2 * d
            if L <= R:
                diff[L] += val
                if R + 1 < N:
                    diff[R + 1] -= val
            else:
                # Split into [L, N-1] and [0, R]
                diff[L] += val
                diff[N] -= val
                diff[0] += val
                if R + 1 < N:
                    diff[R + 1] -= val
        elif ccw < cw:
            # Counter-clockwise path: [b, (a-1) % N]
            L = b
            R = (a - 1) % N
            val = N - 2 * d
            if L <= R:
                diff[L] += val
                if R + 1 < N:
                    diff[R + 1] -= val
            else:
                diff[L] += val
                diff[N] -= val
                diff[0] += val
                if R + 1 < N:
                    diff[R + 1] -= val
    
    # Compute prefix sums to get sum_bridge for each bridge
    sum_bridge = [0] * N
    current = 0
    for i in range(N):
        current += diff[i]
        sum_bridge[i] = current
    
    min_val = min(sum_bridge) if sum_bridge else 0
    print(S + min_val)

if __name__ == '__main__':
    main()