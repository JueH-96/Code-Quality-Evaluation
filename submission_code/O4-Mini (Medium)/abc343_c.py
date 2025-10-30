def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())

    # Compute integer cube root: largest x such that x^3 <= N
    lo, hi = 0, 10**6 + 2
    while lo < hi:
        mid = (lo + hi) // 2
        if mid**3 <= N:
            lo = mid + 1
        else:
            hi = mid
    x = lo - 1

    # Iterate down from x to 1, find the first palindromic cube
    for i in range(x, 0, -1):
        c = i**3
        s = str(c)
        if s == s[::-1]:
            print(c)
            return

if __name__ == "__main__":
    main()