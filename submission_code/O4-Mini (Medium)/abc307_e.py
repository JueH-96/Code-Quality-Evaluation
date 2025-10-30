import sys
import threading

def main():
    import sys

    mod = 998244353
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])

    # We need the number of proper colorings of a cycle of length N
    # with M colors, i.e., adjacent vertices get different colors.
    # Standard formula: (M-1)^N + (-1)^N * (M-1), all mod 998244353.
    # This counts the number of ways to color a cycle graph.
    
    # Compute (M-1)^N mod
    m1 = (M - 1) % mod
    p = pow(m1, N, mod)
    
    # Depending on parity of N, add or subtract (M-1)
    if N % 2 == 0:
        ans = p + m1
    else:
        ans = p - m1
    
    ans %= mod
    print(ans)

if __name__ == "__main__":
    main()