def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    H = list(map(int, data[2:]))

    remaining = M
    count = 0
    for i in range(N):
        if H[i] <= remaining:
            count += 1
            remaining -= H[i]
        else:
            remaining = 0
    print(count)

if __name__ == "__main__":
    main()