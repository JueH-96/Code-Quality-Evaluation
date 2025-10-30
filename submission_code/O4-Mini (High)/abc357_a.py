def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    count = 0
    for _ in range(n):
        h = int(next(it))
        if m >= h:
            count += 1
            m -= h
        else:
            break
    print(count)

if __name__ == "__main__":
    main()