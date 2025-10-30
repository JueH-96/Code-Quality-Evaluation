def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    pieces = data[2:]
    
    R = set()
    C = set()
    S = set()  # i + j
    D = set()  # i - j
    
    for k in range(M):
        a = int(pieces[2*k])
        b = int(pieces[2*k+1])
        R.add(a)
        C.add(b)
        S.add(a + b)
        D.add(a - b)
    
    A = sorted(list(set(range(1, N+1)) - R))
    B = sorted(list(set(range(1, N+1)) - C))
    
    # Compute |X|: number of (i,j) where i in A, j in B, i+j in S
    X = 0
    for s in S:
        for i in A:
            j = s - i
            if 1 <= j <= N and j in B:
                X += 1
    
    # Compute |Y|: number of (i,j) where i in A, j in B, i-j in D
    Y = 0
    for d in D:
        for i in A:
            j = i - d
            if 1 <= j <= N and j in B:
                Y += 1
    
    # Compute |X ∩ Y|: number of (i,j) where i in A, j in B, i+j in S, i-j in D
    X_inter_Y = 0
    for s in S:
        for d in D:
            i = (s + d) / 2
            j = (s - d) / 2
            if i.is_integer() and j.is_integer():
                i = int(i)
                j = int(j)
                if i in A and j in B and 1 <= i <= N and 1 <= j <= N:
                    X_inter_Y += 1
    
    # Number of safe squares
    safe_squares = (N - len(R)) * (N - len(C)) - (X + Y - X_inter_Y)
    print(safe_squares)

if __name__ == '__main__':
    main()