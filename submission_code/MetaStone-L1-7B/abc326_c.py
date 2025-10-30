import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    m = int(data[1])
    a = list(map(int, data[2:2+n]))
    a.sort()
    max_count = 0

    for i in range(n):
        current = a[i]
        upper = current + m
        j = bisect.bisect_left(a, upper, i, n)
        count = j - i
        if count > max_count:
            max_count = count

    print(max_count)

if __name__ == "__main__":
    main()