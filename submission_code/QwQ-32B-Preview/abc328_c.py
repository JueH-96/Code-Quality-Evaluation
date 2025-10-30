def main():
    import sys
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    S = data[ptr]
    ptr += 1
    queries = []
    for _ in range(Q):
        l = int(data[ptr])
        ptr += 1
        r = int(data[ptr])
        ptr += 1
        queries.append((l, r))
    
    # Precompute 'same' array
    same = [0] * (N - 1)
    for i in range(N - 1):
        same[i] = 1 if S[i] == S[i + 1] else 0
    
    # Precompute prefix sum array
    prefix = [0] * N
    for i in range(1, N):
        prefix[i] = prefix[i - 1] + same[i - 1]
    
    # Answer each query
    for l, r in queries:
        if l >= r:
            print(0)
        else:
            print(prefix[r - 1] - prefix[l - 1])

if __name__ == '__main__':
    main()