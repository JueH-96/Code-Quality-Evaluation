import math
import sys

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    gcd = math.gcd(N, M)
    lcm = (N * M) // gcd
    low = 1
    high = K * max(N, M)
    
    def count(x):
        return x // N + x // M - 2 * (x // lcm)
    
    while low < high:
        mid = (low + high) // 2
        c = count(mid)
        if c < K:
            low = mid + 1
        else:
            high = mid
    print(low)

if __name__ == "__main__":
    main()