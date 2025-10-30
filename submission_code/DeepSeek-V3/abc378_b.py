# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    
    qr = []
    for _ in range(N):
        q = int(data[idx])
        r = int(data[idx+1])
        qr.append((q, r))
        idx += 2
    
    Q = int(data[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        t = int(data[idx])
        d = int(data[idx+1])
        queries.append((t, d))
        idx += 2
    
    for t, d in queries:
        q, r = qr[t-1]
        if d % q == r:
            print(d)
            continue
        # Find the smallest day >= d where day % q == r
        # day = k * q + r >= d
        # k >= (d - r) / q
        # So k = ceil((d - r) / q)
        # But since d % q != r, we need to find the next k
        # So k = floor((d - r) / q) + 1
        k = (d - r) // q
        if (d - r) % q != 0:
            k += 1
        next_day = k * q + r
        print(next_day)

if __name__ == "__main__":
    main()