import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    q = int(data[idx])
    idx += 1
    r = list(map(int, data[idx:idx+n]))
    idx += n
    r.sort()
    prefix = [0]
    current = 0
    for num in r:
        current += num
        prefix.append(current)
    for _ in range(q):
        x = int(data[idx])
        idx += 1
        m = bisect.bisect_right(prefix, x) - 1
        print(m)

if __name__ == "__main__":
    main()