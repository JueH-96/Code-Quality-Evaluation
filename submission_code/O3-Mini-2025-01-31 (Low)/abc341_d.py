def main():
    import sys
    input_line = sys.stdin.read().strip()
    if not input_line:
        return
    N, M, K = map(int, input_line.split())
    
    from math import gcd
    lcm = N * M // gcd(N, M)
    
    def count(x):
        # count numbers <= x divisible by exactly one of N or M:
        # Count numbers divisible by N: x // N
        # Count numbers divisible by M: x // M
        # Subtract numbers divisible by both: 2*(x // lcm)
        return x // N + x // M - 2*(x // lcm)
    
    lo, hi = 1, 10**18
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= K:
            hi = mid
        else:
            lo = mid + 1
    print(lo)

if __name__ == '__main__':
    main()