def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    rules = []
    index = 1
    for i in range(n):
        q = int(data[index])
        r = int(data[index+1])
        index += 2
        rules.append((q, r))
    
    Q = int(data[index])
    index += 1
    queries = []
    for i in range(Q):
        t = int(data[index])
        d = int(data[index+1])
        index += 2
        queries.append((t, d))
    
    results = []
    for t, d in queries:
        q, r = rules[t-1]
        if d <= r:
            results.append(str(r))
        else:
            base = d - r
            k_min = (base + q - 1) // q
            ans = k_min * q + r
            results.append(str(ans))
    
    print("
".join(results))

if __name__ == "__main__":
    main()