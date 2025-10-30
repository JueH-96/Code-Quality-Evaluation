def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    total_pairs = n * (n - 1) // 2
    adjacent_set = set()
    index = 2
    for _ in range(m):
        arr = list(map(int, data[index:index + n]))
        index += n
        for j in range(n - 1):
            u = arr[j]
            v = arr[j + 1]
            if u > v:
                u, v = v, u
            adjacent_set.add((u, v))
    print(total_pairs - len(adjacent_set))

if __name__ == '__main__':
    main()