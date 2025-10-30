def main():
    # Parse input
    N, M = map(int, input().split())
    
    # Read grid S
    S = []
    for _ in range(N):
        row = input().strip()
        S.append(row)
    
    # Read grid T
    T = []
    for _ in range(M):
        row = input().strip()
        T.append(row)
    
    # Check all possible positions where T can fit in S
    for a in range(N - M + 1):
        for b in range(N - M + 1):
            match = True
            for i in range(M):
                for j in range(M):
                    if S[a + i][b + j] != T[i][j]:
                        match = False
                        break
                if not match:
                    break
            
            if match:
                # Output using 1-indexed coordinates
                print(a + 1, b + 1)
                return

if __name__ == "__main__":
    main()