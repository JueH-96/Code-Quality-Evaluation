import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    qr = [tuple(map(int, input().split())) for _ in range(N)]
    Q = int(input().strip())
    res = []
    for _ in range(Q):
        t, d = map(int, input().split())
        q, r = qr[t-1]
        mod = d % q
        # compute minimal non-negative shift so that (d + shift) % q == r
        shift = (r - mod) % q
        res.append(str(d + shift))
    sys.stdout.write("
".join(res))

if __name__ == "__main__":
    main()