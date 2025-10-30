def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    # Binary search for floor(cuberoot(N))
    lo, hi = 0, 10**6 + 1  # since (10^6)^3 = 10^18
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid * mid <= N:
            lo = mid
        else:
            hi = mid
    # lo is now floor(cuberoot(N))
    for x in range(lo, 0, -1):
        k = x * x * x
        s = str(k)
        if s == s[::-1]:
            print(k)
            return

# Call main to execute
main()