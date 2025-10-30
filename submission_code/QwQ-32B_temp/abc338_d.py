import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    sum_d = 0
    delta = [0] * (N + 2)  # 1-based to N, delta[N+1] is for overflow

    for i in range(M - 1):
        a = X[i]
        b = X[i + 1]

        # Compute clockwise and counter distances
        if a <= b:
            cw = b - a
        else:
            cw = b + N - a
        counter = N - cw
        d = min(cw, counter)
        sum_d += d
        w = N - 2 * d

        if cw <= counter:
            # Clockwise direction
            if a <= b:
                L = a
                R = b - 1
                delta[L] += w
                delta[R + 1] -= w
            else:
                # First interval [a, N]
                L1 = a
                R1 = N
                delta[L1] += w
                delta[R1 + 1] -= w
                # Second interval [1, b-1]
                L2 = 1
                R2 = b - 1
                if R2 >= L2:
                    delta[L2] += w
                    delta[R2 + 1] -= w
        else:
            # Counter-clockwise direction
            L = b
            R = a - 1
            if L <= R:
                delta[L] += w
                delta[R + 1] -= w
            else:
                # Split into two intervals
                # First [L, N]
                L1 = L
                R1 = N
                delta[L1] += w
                delta[R1 + 1] -= w
                # Second [1, R]
                L2 = 1
                R2 = R
                if R2 >= L2:
                    delta[L2] += w
                    delta[R2 + 1] -= w

    # Compute prefix sum to get contributions
    current = 0
    min_contribution = float('inf')
    for i in range(1, N + 1):
        current += delta[i]
        if current < min_contribution:
            min_contribution = current

    ans = sum_d + min_contribution
    print(ans)

if __name__ == "__main__":
    main()