import sys

def main() -> None:
    it = map(int, sys.stdin.buffer.read().split())
    N = next(it)

    A = [next(it) for _ in range(N)]
    B = [next(it) for _ in range(N)]
    C = [next(it) for _ in range(N)]

    S = 0                  # current weighted sum of ones
    down = []              # weights to flip 1 -> 0
    up   = []              # weights to flip 0 -> 1

    for a, b, c in zip(A, B, C):
        if a == 1:
            S += c
        if a == 1 and b == 0:
            down.append(c)
        elif a == 0 and b == 1:
            up.append(c)

    down.sort(reverse=True)    # descending
    up.sort()                  # ascending

    total = 0

    # perform all down flips
    for c in down:
        S -= c         # bit becomes 0
        total += S

    # perform all up flips
    for c in up:
        S += c         # bit becomes 1
        total += S

    print(total)

if __name__ == "__main__":
    main()