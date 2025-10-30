import bisect

def main():
    import sys
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    total = 0
    for j in range(1, n):
        target = A[j] / 2
        idx = bisect.bisect_right(A, target, 0, j)
        total += idx
    print(total)

if __name__ == "__main__":
    main()