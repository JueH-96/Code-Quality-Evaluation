def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:1+N]))
    Q = list(map(int, data[1+N:1+2*N]))
    
    # Convert P, Q into zero-based "ranks" so that:
    #   rowRank[i] = the position of row i in the order P (0-based)
    #   colRank[j] = the position of column j in the order Q (0-based)
    # Example: if P = [2,1,3], then rowRank[1] = 1 (row #1 is in position 1)
    #                                rowRank[2] = 0
    #                                rowRank[3] = 2
    rowRank = [0]*N
    colRank = [0]*N
    for i in range(N):
        rowRank[P[i]-1] = i
        colRank[Q[i]-1] = i
    
    # We'll fill an N-by-N grid with '0' or '1' by the rule:
    #   grid[i][j] = '1' if rowRank[i] > colRank[j], else '0'.
    # This ensures S_{P_1}<...<S_{P_N} and T_{Q_1}<...<T_{Q_N} in lexicographic order.
    result = []
    for i in range(N):
        row_i = []
        for j in range(N):
            if rowRank[i] > colRank[j]:
                row_i.append('1')
            else:
                row_i.append('0')
        result.append("".join(row_i))
    
    # Print the result
    print("
".join(result))

# Don't forget to call main!
main()