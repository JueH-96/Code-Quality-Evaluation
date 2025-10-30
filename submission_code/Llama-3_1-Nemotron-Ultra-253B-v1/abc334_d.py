import bisect
import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    r.sort()
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + r[i - 1]
    for _ in range(q):
        x = int(sys.stdin.readline())
        ans = bisect.bisect_right(prefix, x) - 1
        print(ans)

if __name__ == "__main__":
    main()