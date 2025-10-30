def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    Q = list(map(int, data[1+n:1+2*n]))
    
    rank_row = [0] * (n + 1)
    for idx in range(n):
        row_num = P[idx]
        rank_row[row_num] = idx
        
    rank_col = [0] * (n + 1)
    for idx in range(n):
        col_num = Q[idx]
        rank_col[col_num] = idx
        
    output_lines = []
    for i in range(n):
        s = []
        for j in range(n):
            total_rank = rank_row[i + 1] + rank_col[j + 1]
            if total_rank >= n - 1:
                s.append('1')
            else:
                s.append('0')
        output_lines.append(''.join(s))
    
    print('
'.join(output_lines))

if __name__ == "__main__":
    main()