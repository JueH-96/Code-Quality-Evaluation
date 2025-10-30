import bisect
import sys

def main():
    n = int(sys.stdin.readline())
    x = list(map(int, sys.stdin.readline().split()))
    q = int(sys.stdin.readline())
    tg = [list(map(int, sys.stdin.readline().split())) for _ in range(q)]
    ans = 0
    for t, g in tg:
        t -= 1
        idx = bisect.bisect_left(x, g)
        if idx == 0:
            ans += abs(x[t] - g)
            x[t] = g
        elif idx == n:
            ans += abs(x[t] - x[-1]) + abs(x[-1] - g)
            for i in range(t, n):
                x[i] = x[-1]
            x[-1] = g
        else:
            d = min(abs(x[idx - 1] - g), abs(x[idx] - g))
            if x[t] <= g:
                ans += abs(x[t] - x[idx - 1]) + d
                for i in range(t, idx):
                    x[i] = x[idx - 1]
                x[idx - 1] = g
            else:
                ans += abs(x[t] - x[idx]) + d
                for i in range(t, idx + 1):
                    x[i] = x[idx]
                x[idx] = g
    print(ans)

if __name__ == "__main__":
    main()