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
        queries.append((t-1, d))  # converting to 0-based index
        idx += 2
    
    for t, d in queries:
        q, r = qr[t]
        if d % q == r:
            print(d)
            continue
        # Find the smallest x >= d such that x % q == r
        # x = d + (r - d % q) % q
        remainder = d % q
        if remainder <= r:
            x = d + (r - remainder)
        else:
            x = d + (q - remainder + r)
        print(x)

if __name__ == "__main__":
    main()