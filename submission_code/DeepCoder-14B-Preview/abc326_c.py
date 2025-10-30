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
        target = a[i] + m
        pos = bisect.bisect_left(a, target)
        j = pos - 1
        current = j - i + 1
        if current > max_count:
            max_count = current
    print(max_count)

if __name__ == '__main__':
    main()