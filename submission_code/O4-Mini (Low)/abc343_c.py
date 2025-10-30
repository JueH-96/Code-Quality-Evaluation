def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)

    # Find integer floor of cube root of N via binary search
    lo, hi = 0, 10**6 + 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**3 <= N:
            lo = mid
        else:
            hi = mid - 1
    # lo is the largest x with x^3 <= N
    for x in range(lo, 0, -1):
        c = x**3
        s = str(c)
        if s == s[::-1]:
            print(c)
            return

if __name__ == "__main__":
    main()