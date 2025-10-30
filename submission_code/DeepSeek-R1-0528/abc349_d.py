def main():
    import sys
    data = sys.stdin.read().split()
    L = int(data[0])
    R = int(data[1])
    segments = []
    current = L
    while current < R:
        if current == 0:
            d = 1
            while d * 2 <= R:
                d *= 2
            segments.append((current, current + d))
            current += d
        else:
            d0 = current & -current
            d = d0
            while d > R - current:
                d //= 2
            segments.append((current, current + d))
            current += d
    print(len(segments))
    for a, b in segments:
        print(a, b)

if __name__ == "__main__":
    main()