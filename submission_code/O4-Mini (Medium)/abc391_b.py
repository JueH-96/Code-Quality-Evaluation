def main():
    import sys
    input = sys.stdin.readline
    
    # Read dimensions
    N, M = map(int, input().split())
    
    # Read the big grid S
    S = [input().rstrip('
') for _ in range(N)]
    # Read the small grid T
    T = [input().rstrip('
') for _ in range(M)]
    
    # Try every possible top-left corner (a, b) of an M×M subgrid in S
    # Note: in 0-indexed Python, these go from 0 to N-M
    for a in range(N - M + 1):
        for b in range(N - M + 1):
            match = True
            # Check if the subgrid starting at (a, b) matches T
            for i in range(M):
                # Compare the substring of S at row (a+i), columns b..b+M-1 to T[i]
                if S[a + i][b:b + M] != T[i]:
                    match = False
                    break
            if match:
                # Convert to 1-based indexing and print
                print(a + 1, b + 1)
                return

if __name__ == "__main__":
    main()