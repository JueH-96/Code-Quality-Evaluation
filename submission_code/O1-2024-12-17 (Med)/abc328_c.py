def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    S = data[2]
    
    # Precompute an array B where B[i] = 1 if S[i] == S[i+1], else 0
    B = [0]*(N-1)
    for i in range(N-1):
        B[i] = 1 if S[i] == S[i+1] else 0
    
    # Compute prefix sums P, where P[i] = sum of B[0..i-1]
    P = [0]*N
    for i in range(1, N):
        P[i] = P[i-1] + B[i-1]
    
    # Process the queries
    idx = 3
    answers = []
    for _ in range(Q):
        l, r = map(int, data[idx:idx+2])
        idx += 2
        # The number of positions where S[p] = S[p+1] in S[l..r]
        # is the sum of B[p] for p in [l-1..r-2], i.e. P[r-1] - P[l-1].
        if r == l:
            answers.append('0')
        else:
            answers.append(str(P[r-1] - P[l-1]))
    
    # Print all results
    print('
'.join(answers))

# Do not forget to call main()!
if __name__ == "__main__":
    main()