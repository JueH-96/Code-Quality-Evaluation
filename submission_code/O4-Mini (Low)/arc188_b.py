import sys

def main():
    input = sys.stdin.readline
    t = int(input())
    out = []
    for _ in range(t):
        N, K = map(int, input().split())
        # The only impossible case is when Bob is exactly opposite Alice,
        # i.e. 2*K ≡ 0 (mod N), and thus K = N/2 (N even).
        if (2*K) % N == 0:
            out.append("No")
        else:
            out.append("Yes")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()