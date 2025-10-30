import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    a = list(map(int, data[1:n+1]))
    a.sort()
    total = 0
    for i in range(n-1):
        x = a[i]
        left = i + 1
        right = n - 1
        current = left
        s = 0
        while current <= right:
            m = a[current] // x
            target = (m + 1) * x
            next_pos = bisect.bisect_left(a, target, current, right + 1)
            count = next_pos - current
            s += m * count
            current = next_pos
        total += s
    print(total)

if __name__ == "__main__":
    main()