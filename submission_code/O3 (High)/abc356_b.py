import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    it = iter(data)
    N = next(it)
    M = next(it)

    # Required amounts per nutrient
    A = [next(it) for _ in range(M)]

    # Accumulate nutrients actually taken
    taken = [0] * M
    for _ in range(N):
        for j in range(M):
            taken[j] += next(it)

    # Check if every nutrient meets or exceeds the target
    ok = all(taken[j] >= A[j] for j in range(M))

    print("Yes" if ok else "No")

if __name__ == "__main__":
    main()