import sys

def main() -> None:
    # Read all input at once for speed
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    A = list(map(int, data[1:1 + N]))

    # Prefix distinct counts
    prefix = [0] * N
    seen = set()
    for i, x in enumerate(A):
        seen.add(x)
        prefix[i] = len(seen)

    # Suffix distinct counts
    suffix = [0] * N
    seen.clear()
    for i in range(N - 1, -1, -1):
        seen.add(A[i])
        suffix[i] = len(seen)

    # Evaluate every split position
    best = 0
    for i in range(N - 1):           # split between i and i+1
        best = max(best, prefix[i] + suffix[i + 1])

    print(best)

if __name__ == "__main__":
    main()