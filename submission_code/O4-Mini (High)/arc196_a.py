import sys
import threading

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    A.sort()
    half = n // 2
    res = 0
    # Pair the largest half with the smallest half
    # i.e. A[n-1] with A[0], A[n-2] with A[1], etc.
    for i in range(half):
        res += A[n-1-i] - A[i]
    print(res)

if __name__ == "__main__":
    main()