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

    sum_d = 0
    diff = [0] * (N + 2)  # 1-based to N, N+1 for overflow

    for i in range(M-1):
        x = X[i]
        y = X[i+1]

        d_cw = (y - x) % N
        d_ccw = (x - y) % N

        if d_cw < d_ccw:
            L = x
            if y == 1:
                R = N
            else:
                R = y - 1

            delta_i = N - 2 * d_cw

            if L <= R:
                diff[L] += delta_i
                if R + 1 <= N:
                    diff[R + 1] -= delta_i
                else:
                    diff[N + 1] -= delta_i
            else:
                diff[L] += delta_i
                diff[N + 1] -= delta_i
                if R >= 1:
                    diff[1] += delta_i
                    if R + 1 <= N:
                        diff[R + 1] -= delta_i
                    else:
                        diff[N + 1] -= delta_i
        else:
            L = y
            if x == 1:
                R = N
            else:
                R = x - 1

            delta_i = N - 2 * d_ccw

            if L <= R:
                diff[L] += delta_i
                if R + 1 <= N:
                    diff[R + 1] -= delta_i
                else:
                    diff[N + 1] -= delta_i
            else:
                diff[L] += delta_i
                diff[N + 1] -= delta_i
                if R >= 1:
                    diff[1] += delta_i
                    if R + 1 <= N:
                        diff[R + 1] -= delta_i
                    else:
                        diff[N + 1] -= delta_i

        sum_d += min(d_cw, d_ccw)

    # Compute prefix sum to get delta_total for each bridge a
    delta_total = [0] * (N + 2)
    current = 0
    for a in range(1, N + 1):
        current += diff[a]
        delta_total[a] = current

    # Find the minimal sum_total
    min_total = sum_d + min(delta_total[1:N+1])
    print(min_total)

if __name__ == '__main__':
    main()