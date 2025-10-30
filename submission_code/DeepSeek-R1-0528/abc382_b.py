def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, d = map(int, data[0].split())
    s = data[1].strip()
    res = list(s)
    count = 0
    for i in range(n-1, -1, -1):
        if count == d:
            break
        if res[i] == '@':
            res[i] = '.'
            count += 1
    print(''.join(res))

if __name__ == '__main__':
    main()