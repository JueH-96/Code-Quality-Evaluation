def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    counts = {}
    index = 1
    for _ in range(N):
        S = int(data[index])
        C = int(data[index + 1])
        counts[S] = C
        index += 2
    
    sizes = sorted(counts.keys())
    i = 0
    while i < len(sizes):
        S = sizes[i]
        while counts[S] >= 2:
            pairs = counts[S] // 2
            remainder = counts[S] % 2
            counts[S] = remainder
            next_size = 2 * S
            counts[next_size] += pairs
            if next_size not in sizes:
                sizes.append(next_size)
        i += 1
    
    total_slimes = sum(counts.values())
    print(total_slimes)

if __name__ == '__main__':
    main()