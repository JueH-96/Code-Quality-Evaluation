def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    S = data[1+N]
    
    # 1) Build prefix arrays for 'M'
    # prefixM[x][i] = number of positions up to (but not including) i
    # such that S[pos] = 'M' and A[pos] = x.
    prefixM = [[0]*(N+1) for _ in range(3)]
    for i in range(N):
        for x in range(3):
            prefixM[x][i+1] = prefixM[x][i]
        if S[i] == 'M':
            prefixM[A[i]][i+1] += 1

    # 2) Build suffix arrays for 'X'
    # suffixX[x][i] = number of positions from i to the end
    # such that S[pos] = 'X' and A[pos] = x.
    suffixX = [[0]*(N+1) for _ in range(3)]
    for x in range(3):
        suffixX[x][N] = 0
    for i in range(N-1, -1, -1):
        for x in range(3):
            suffixX[x][i] = suffixX[x][i+1]
        if S[i] == 'X':
            suffixX[A[i]][i] += 1

    # 3) Precompute the mex for all (x, y, z) in {0,1,2}
    # mex is the smallest non-negative integer not in {x,y,z}.
    mex_table = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for x in range(3):
        for y in range(3):
            for z in range(3):
                vals = {x, y, z}
                if 0 not in vals:
                    mex_table[x][y][z] = 0
                elif 1 not in vals:
                    mex_table[x][y][z] = 1
                elif 2 not in vals:
                    mex_table[x][y][z] = 2
                else:
                    mex_table[x][y][z] = 3

    # 4) Sum up mex(A_i, A_j, A_k) for i<j<k with S_iS_jS_k = 'MEX'.
    #    For each j where S[j] = 'E', sum over all i < j in M positions, k > j in X positions.
    answer = 0
    for j in range(N):
        if S[j] == 'E':
            y = A[j]
            for x in range(3):
                countM = prefixM[x][j]     # how many M's with A=? up to j
                if countM > 0:
                    for z in range(3):
                        countX = suffixX[z][j+1]  # how many X's with A=? after j
                        if countX > 0:
                            answer += mex_table[x][y][z] * countM * countX

    print(answer)

# Don't forget to call main!
if __name__ == "__main__":
    main()