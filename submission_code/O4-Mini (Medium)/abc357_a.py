def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    hands = list(map(int, data[2:]))

    count = 0
    remaining = M
    for h in hands:
        if remaining >= h:
            count += 1
            remaining -= h
        else:
            break

    print(count)

if __name__ == '__main__':
    main()