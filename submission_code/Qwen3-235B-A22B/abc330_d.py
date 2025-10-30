def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = []
    for _ in range(N):
        S.append(input[idx])
        idx += 1
    
    row_counts = [row.count('o') for row in S]
    
    column_counts = [0] * N
    for j in range(N):
        cnt = 0
        for i in range(N):
            if S[i][j] == 'o':
                cnt += 1
        column_counts[j] = cnt
    
    total = 0
    for i in range(N):
        for j in range(N):
            if S[i][j] == 'o':
                total += (row_counts[i] - 1) * (column_counts[j] - 1)
    
    print(total)

if __name__ == '__main__':
    main()