import bisect
import sys

def main():
    n = int(sys.stdin.readline())
    x = list(map(int, sys.stdin.readline().split()))
    p = list(map(int, sys.stdin.readline().split()))
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + p[i]
    q = int(sys.stdin.readline())
    for _ in range(q):
        L, R = map(int, sys.stdin.readline().split())
        a = bisect.bisect_left(x, L)
        b = bisect.bisect_right(x, R) - 1
        if a > b:
            print(0)
        else:
            print(prefix[b + 1] - prefix[a])

if __name__ == "__main__":
    main()