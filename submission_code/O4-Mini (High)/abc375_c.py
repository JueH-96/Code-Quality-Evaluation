import sys
def main():
    input = sys.stdin.readline
    N = int(input())
    A = [ input().rstrip() for _ in range(N) ]
    m = N - 1
    # Precompute dj = min(j, m-j) for j in [0..N-1]
    djs = [ j if j <= m-j else m-j for j in range(N) ]
    write = sys.stdout.write

    for i in range(N):
        # distance of this row to the border
        di = i if i <= m - i else m - i
        Ai = A[i]
        Ami = A[m - i]
        # build output row
        row = [''] * N
        for j in range(N):
            # distance of this column to border
            dj = djs[j]
            # layer index = min(di, dj)
            d = dj if dj < di else di
            # number of 90° cw rotations mod 4
            r = (d + 1) & 3

            if r == 0:
                # no rotation
                row[j] = Ai[j]
            elif r == 1:
                # 90° cw  => original came from (m-j, i)
                row[j] = A[m - j][i]
            elif r == 2:
                # 180°    => original came from (m-i, m-j)
                row[j] = Ami[m - j]
            else:
                # 270° cw => original came from (j, m-i)
                row[j] = A[j][m - i]

        # write this row
        write(''.join(row))
        write('
')

if __name__ == "__main__":
    main()