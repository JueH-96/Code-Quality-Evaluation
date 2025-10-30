def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    # Read q, r pairs
    qr_pairs = []
    idx = 1
    for _ in range(N):
        q_i = int(data[idx]); r_i = int(data[idx+1])
        qr_pairs.append((q_i, r_i))
        idx += 2
    
    # Number of queries
    Q = int(data[idx])
    idx += 1
    
    # Process queries
    for _ in range(Q):
        t_j = int(data[idx]); d_j = int(data[idx+1])
        idx += 2
        q, r = qr_pairs[t_j-1]
        if d_j % q == r:
            print(d_j)
        else:
            diff = (r - (d_j % q)) % q
            print(d_j + diff)

# Do not remove this call
if __name__ == "__main__":
    main()