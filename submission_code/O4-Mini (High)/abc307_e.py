import sys
def main():
    input = sys.stdin.readline
    mod = 998244353

    N, M = map(int, input().split())
    # We want the number of proper M-colorings of a cycle of length N:
    # Chromatic polynomial of cycle C_N: (M-1)^N + (-1)^N * (M-1)
    x = M - 1
    p = pow(x, N, mod)
    if N % 2 == 0:
        ans = (p + x) % mod
    else:
        ans = (p - x) % mod
    print(ans)

if __name__ == "__main__":
    main()