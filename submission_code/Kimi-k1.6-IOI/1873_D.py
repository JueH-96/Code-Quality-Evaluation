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
        k = int(data[idx+1])
        idx +=2
        s = data[idx]
        idx +=1
        bs = [i for i, c in enumerate(s) if c == 'B']
        if not bs:
            results.append(0)
            continue
        count = 0
        last_end = -1
        for pos in bs:
            if pos > last_end:
                count +=1
                last_end = pos + k -1
        results.append(count)
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()