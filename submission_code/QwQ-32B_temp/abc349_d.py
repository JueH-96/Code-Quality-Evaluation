def main():
    import sys
    L, R = map(int, sys.stdin.readline().split())
    current = L
    segments = []
    while current < R:
        if current == 0:
            n = R - current
            if n == 0:
                step = 0
            else:
                step = 1 << (n.bit_length() - 1)
        else:
            max_power = current & -current
            candidate = max_power
            while candidate > (R - current):
                candidate //= 2
            step = candidate
        next_current = current + step
        segments.append((current, next_current))
        current = next_current
    print(len(segments))
    for l, r in segments:
        print(l, r)

if __name__ == "__main__":
    main()