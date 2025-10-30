def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    S = []
    T = []
    
    index = 2
    for i in range(N):
        S.append(data[index])
        index += 1
    
    for i in range(M):
        T.append(data[index])
        index += 1
    
    # We need to find the top-left corner (a, b) such that the subgrid S[a:a+M][b:b+M] matches T
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
                print(a + 1, b + 1)
                return

if __name__ == "__main__":
    main()