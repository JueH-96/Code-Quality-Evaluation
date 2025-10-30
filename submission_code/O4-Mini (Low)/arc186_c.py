import sys
import threading

def main():
    input = sys.stdin.readline
    T = int(input())
    out = []
    for _ in range(T):
        line = input().split()
        if not line:
            line = input().split()
        N, M = map(int, line)
        diffs = []
        for _ in range(N):
            v, p = map(int, input().split())
            d = v - p
            if d > 0:
                diffs.append(d)
        # We only care about the top M positive differences
        if not diffs:
            out.append("0")
            continue
        diffs.sort(reverse=True)
        k = min(M, len(diffs))
        ans = sum(diffs[:k])
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()