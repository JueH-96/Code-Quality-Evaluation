def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    # First line contains two integers: N and M
    first_line = input_data[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    
    # Next N lines contain grid S, following M lines contain grid T
    S = input_data[1:1+N]
    T = input_data[1+N:1+N+M]
    
    # Loop over all possible starting positions in S for the MxM subgrid
    for a in range(0, N - M + 1):
        for b in range(0, N - M + 1):
            found = True
            for i in range(M):
                if S[a + i][b:b + M] != T[i]:
                    found = False
                    break
            if found:
                # Output using 1-indexed positions
                print(a + 1, b + 1)
                return

if __name__ == '__main__':
    main()