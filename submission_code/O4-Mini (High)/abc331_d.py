import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N, Q = map(int, line)
    P = [data.readline().rstrip() for _ in range(N)]
    # build 2D prefix sum S of P
    # S[i][j] = number of 'B' in submatrix P[0..i-1][0..j-1]
    S = [[0] * (N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        row = P[i-1]
        Si = S[i]
        Sim1 = S[i-1]
        cum = 0
        for j in range(1, N+1):
            # cum = sum over row[i-1][0..j-1]
            if row[j-1] == 'B':
                cum += 1
            # S[i][j] = S[i-1][j] + cum
            Si[j] = Sim1[j] + cum

    total = S[N][N]

    def f(x, y):
        # count blacks in rectangle [0..x] x [0..y]
        if x < 0 or y < 0:
            return 0
        # number of full N-blocks in rows / remainder
        a, r = divmod(x+1, N)
        # number of full N-blocks in cols / remainder
        b, c = divmod(y+1, N)
        # sum of full blocks + vertical strip + horizontal strip + corner
        return a*b*total + a * S[N][c] + b * S[r][N] + S[r][c]

    out = []
    append = out.append
    for _ in range(Q):
        A, B, C, D = map(int, data.readline().split())
        res = f(C, D) - f(A-1, D) - f(C, B-1) + f(A-1, B-1)
        append(str(res))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()