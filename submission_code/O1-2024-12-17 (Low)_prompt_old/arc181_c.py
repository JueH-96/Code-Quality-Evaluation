def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:1+N]))
    Q = list(map(int, data[1+N:1+2*N]))

    # rankRow[i] = r  means: row i has rank r in the permutation P
    # i.e. P[r-1] = i   (because P is 1-indexed in the problem statement)
    rankRow = [0]*(N+1)
    for r in range(1, N+1):
        row_index = P[r-1]
        rankRow[row_index] = r

    # rankCol[j] = c  means: column j has rank c in the permutation Q
    rankCol = [0]*(N+1)
    for c in range(1, N+1):
        col_index = Q[c-1]
        rankCol[col_index] = c

    # We'll fill the grid A of size N×N with characters '0' or '1'.
    # A_{i,j} = '1' if rankRow[i] + rankCol[j] >= (N+1), else '0'.
    # This construction guarantees that the rows and columns
    # in the order of P and Q are strictly increasing lexicographically.

    # Build and output the result
    # Note: rows and columns are both numbered 1..N.
    # We'll output row i in the order j=1..N.
    result = []
    for i in range(1, N+1):
        row_chars = []
        for j in range(1, N+1):
            if rankRow[i] + rankCol[j] >= N+1:
                row_chars.append('1')
            else:
                row_chars.append('0')
        result.append("".join(row_chars))

    print("
".join(result))

def main():
    solve()

if __name__ == "__main__":
    main()