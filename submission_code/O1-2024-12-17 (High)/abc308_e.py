def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    S = data[1+N]

    # 1) Build prefix counts for 'M'
    # prefixM[i][v] = how many indices ≤ i have S_i='M' and A_i=v
    prefixM = [[0]*3 for _ in range(N+1)]
    for i in range(N):
        for v in range(3):
            prefixM[i+1][v] = prefixM[i][v]
        if S[i] == 'M':
            prefixM[i+1][A[i]] += 1

    # 2) Build suffix counts for 'X'
    # suffixX[i][v] = how many indices ≥ i have S_i='X' and A_i=v
    suffixX = [[0]*3 for _ in range(N+2)]
    for i in range(N, 0, -1):
        for v in range(3):
            suffixX[i][v] = suffixX[i+1][v]
        if S[i-1] == 'X':
            suffixX[i][A[i-1]] += 1

    # 3) Precompute all possible mex values for triples (a, b, c) with a,b,c in {0, 1, 2}
    mex_table = [[[0]*3 for _ in range(3)] for _ in range(3)]
    for x in range(3):
        for y in range(3):
            for z in range(3):
                s = {x, y, z}
                if 0 not in s:
                    mex_table[x][y][z] = 0
                elif 1 not in s:
                    mex_table[x][y][z] = 1
                elif 2 not in s:
                    mex_table[x][y][z] = 2
                else:
                    mex_table[x][y][z] = 3

    # 4) Calculate the required sum
    # For every j such that S_j='E', sum over all a,c the contribution:
    #   prefixM[j-1][a] * suffixX[j+1][c] * mex_table[a][A_j][c]
    ans = 0
    for j in range(1, N+1):
        if S[j-1] == 'E':
            b = A[j-1]
            for a in range(3):
                for c in range(3):
                    ans += prefixM[j-1][a] * suffixX[j+1][c] * mex_table[a][b][c]

    # 5) Output the result
    print(ans)

# Don't forget to call main()!
if __name__ == "__main__":
    main()