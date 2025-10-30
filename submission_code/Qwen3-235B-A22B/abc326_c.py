import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    a = list(map(int, input[2:2+n]))
    a.sort()
    max_count = 0
    for i in range(n):
        target = a[i] + m
        j = bisect.bisect_left(a, target, i)
        current = j - i
        if current > max_count:
            max_count = current
    print(max_count)

if __name__ == "__main__":
    main()