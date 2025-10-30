import sys

def main():
    import bisect
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    P.sort()
    remaining = M
    total = 0
    for p in P:
        if remaining <= 0:
            break
        low = 0
        high = int((remaining / p) ** 0.5) + 1
        best = 0
        while low <= high:
            mid = (low + high) // 2
            cost = mid * mid * p
            if cost <= remaining:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        total += best
        remaining -= best * best * p
    print(total)

if __name__ == "__main__":
    main()