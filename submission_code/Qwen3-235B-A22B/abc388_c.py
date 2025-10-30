import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    a = list(map(int, data[1:]))
    total = 0
    for j in range(1, n):
        x = a[j] / 2.0
        count = bisect.bisect_right(a, x, 0, j)
        total += count
    print(total)

if __name__ == "__main__":
    main()