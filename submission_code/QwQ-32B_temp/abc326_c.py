import bisect

def main():
    import sys
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    a.sort()
    max_count = 0
    for i in range(n):
        target = a[i] + m
        j = bisect.bisect_left(a, target)
        current = j - i
        if current > max_count:
            max_count = current
    print(max_count)

if __name__ == "__main__":
    main()