def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    results = []
    for _ in range(t):
        n = int(data[idx])
        k = int(data[idx + 1])
        idx += 2
        s = data[idx]
        idx += 1
        count = 0
        pos = 0
        while pos < n:
            if s[pos] == 'B':
                count += 1
                pos += k
            else:
                pos += 1
        results.append(count)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()