def main():
    import sys
    # Read N and K
    N_K = sys.stdin.readline().strip()
    while N_K == '':
        N_K = sys.stdin.readline().strip()
    N, K = map(int, N_K.split())
    
    # Read S
    S = sys.stdin.readline().strip()
    while len(S) < N:
        S += sys.stdin.readline().strip()
    S = list(S[:N])  # Ensure S has exactly N characters
    
    counter = 0
    for i in range(N - K + 1):
        if S[i:i+K] == ['O'] * K:
            counter += 1
            S[i:i+K] = ['X'] * K
    print(counter)

if __name__ == "__main__":
    main()