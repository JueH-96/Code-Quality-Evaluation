def compute_steps(X, Y, avoid, N):
    if X == Y:
        return 0
    clockwise = (Y - X) % N
    counter = (X - Y) % N

    # Check if clockwise path includes avoid
    if X < Y:
        cw_includes = X <= avoid <= Y
    else:
        cw_includes = avoid >= X or avoid <= Y

    # Check if counter-clockwise path includes avoid
    if X > Y:
        ccw_includes = Y <= avoid <= X
    else:
        ccw_includes = avoid <= X or avoid >= Y

    if cw_includes:
        return counter
    elif ccw_includes:
        return clockwise
    else:
        return min(clockwise, counter)

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1

    L = 1
    R = 2
    total = 0

    for _ in range(Q):
        H = input[idx]
        idx += 1
        T = int(input[idx])
        idx += 1

        if H == 'L':
            steps = compute_steps(L, T, R, N)
            total += steps
            L = T
        else:
            steps = compute_steps(R, T, L, N)
            total += steps
            R = T

    print(total)

if __name__ == "__main__":
    main()