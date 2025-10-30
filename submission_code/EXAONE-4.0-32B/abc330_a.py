def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    l = int(data[1])
    scores = list(map(int, data[2:2+n]))
    count = 0
    for s in scores:
        if s >= l:
            count += 1
    print(count)

if __name__ == '__main__':
    main()