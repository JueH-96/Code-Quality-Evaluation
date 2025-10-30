def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    # first line: N M
    N, M = map(int, input_data[0].split())
    # Next N lines are grid S, following M lines are grid T.
    S = input_data[1:1+N]
    T = input_data[1+N:1+N+M]
    
    # Iterate over all valid starting positions in S
    for a in range(0, N - M + 1):
        for b in range(0, N - M + 1):
            # Check if subgrid of S matches T
            match = True
            for i in range(M):
                if S[a+i][b:b+M] != T[i]:
                    match = False
                    break
            if match:
                # Print a+1, b+1 (1-indexed)
                print(a+1, b+1)
                return

if __name__ == '__main__':
    main()