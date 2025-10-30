import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute prefix distinct counts
    seen = set()
    prefix = [0] * N
    count = 0
    for i in range(N):
        if A[i] not in seen:
            seen.add(A[i])
            count += 1
        prefix[i] = count

    # Compute suffix distinct counts
    seen.clear()
    suffix = [0] * N
    count = 0
    for i in range(N-1, -1, -1):
        if A[i] not in seen:
            seen.add(A[i])
            count += 1
        suffix[i] = count

    # Try all splits between i and i+1 for i in [0..N-2]
    ans = 0
    for i in range(N-1):
        total = prefix[i] + suffix[i+1]
        if total > ans:
            ans = total

    print(ans)

if __name__ == "__main__":
    main()