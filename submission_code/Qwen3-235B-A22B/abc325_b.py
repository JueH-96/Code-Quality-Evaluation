def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    bases = []
    index = 1
    for _ in range(N):
        w = int(data[index])
        x = int(data[index + 1])
        bases.append((w, x))
        index += 2
    
    max_total = 0
    for S in range(24):
        current = 0
        for w, x in bases:
            local_start = (S + x) % 24
            if 9 <= local_start <= 17:
                current += w
        if current > max_total:
            max_total = current
    print(max_total)

if __name__ == "__main__":
    main()