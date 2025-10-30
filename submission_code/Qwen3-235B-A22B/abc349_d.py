def main():
    import sys
    L, R = map(int, sys.stdin.readline().split())
    intervals = []
    current = L
    while current < R:
        space = R - current
        if current == 0:
            s = 1 << (space.bit_length() - 1)
        else:
            lsb = current & -current
            if lsb <= space:
                s = lsb
            else:
                s = 1 << (space.bit_length() - 1)
        end = current + s
        intervals.append((current, end))
        current = end
    print(len(intervals))
    for l, r in intervals:
        print(l, r)

if __name__ == "__main__":
    main()