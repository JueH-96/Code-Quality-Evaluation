def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return

    # Parse N and M
    N, M = map(int, input_data[0].split())

    # Read grid S and grid T from input
    S = input_data[1:1+N]
    T = input_data[1+N:1+N+M]

    # Iterate over all possible top-left positions for the MxM subgrid in S.
    for a in range(N - M + 1):
        for b in range(N - M + 1):
            # Check if the subgrid of S starting at (a, b) matches grid T.
            is_match = True
            for i in range(M):
                if S[a + i][b:b+M] != T[i]:
                    is_match = False
                    break
            if is_match:
                # Output positions in 1-indexed format
                print(a + 1, b + 1)
                return

if __name__ == '__main__':
    main()