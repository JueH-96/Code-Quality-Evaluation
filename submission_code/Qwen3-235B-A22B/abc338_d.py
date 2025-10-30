import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:]))

    S_original = 0
    diff = [0] * (N + 2)  # diff[1..N], diff[N+1] is also used

    for i in range(M - 1):
        a = X[i]
        b = X[i + 1]
        cw = (b - a) % N
        ccw = (a - b) % N
        d_original = min(cw, ccw)
        S_original += d_original

        if cw == ccw:
            continue

        if cw < ccw:
            L = a
            R = b - 1
            if R < 1:
                R += N
        else:
            L = b
            R = a - 1
            if R < 1:
                R += N

        delta_i = N - 2 * d_original

        if L <= R:
            diff[L] += delta_i
            if R + 1 <= N:
                diff[R + 1] -= delta_i
            else:
                diff[N + 1] -= delta_i
        else:
            # Handle wrap-around
            diff[L] += delta_i
            diff[N + 1] -= delta_i
            diff[1] += delta_i
            if R + 1 <= N:
                diff[R + 1] -= delta_i
            else:
                diff[N + 1] -= delta_i

    # Find the minimal total
    current = 0
    min_total = float('inf')
    for k in range(1, N + 1):
        current += diff[k]
        total = S_original + current
        if total < min_total:
            min_total = total

    print(min_total)

if __name__ == "__main__":
    main()