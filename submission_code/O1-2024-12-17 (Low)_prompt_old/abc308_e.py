def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:1+N]))
    S = input_data[1+N]

    # Precompute mex for all possible triples (a,b,c) where each is in {0,1,2}
    # Since we only need 0,1,2 in A_i, we won't need anything else
    mex_table = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for x in range(3):
        for y in range(3):
            for z in range(3):
                s = {x, y, z}
                # The smallest non-negative integer not in {x,y,z}
                if 0 not in s:
                    mex_table[x][y][z] = 0
                elif 1 not in s:
                    mex_table[x][y][z] = 1
                elif 2 not in s:
                    mex_table[x][y][z] = 2
                else:
                    # next is 3
                    mex_table[x][y][z] = 3

    # We will create prefix arrays for M (since we need i < j, i in M)
    # prefixM[a][i] = number of positions <= i with S[pos]='M' and A[pos]=a
    prefixM = [[0]*(N+1) for _ in range(3)]

    # We will create suffix arrays for X (since we need k > j, k in X)
    # suffixX[a][i] = number of positions >= i with S[pos]='X' and A[pos]=a
    suffixX = [[0]*(N+2) for _ in range(3)]  # +2 to safely handle boundary

    # Build prefixM
    for i in range(1, N+1):
        # copy previous counts
        for a in range(3):
            prefixM[a][i] = prefixM[a][i-1]
        if S[i-1] == 'M':
            prefixM[A[i-1]][i] += 1

    # Build suffixX
    for i in range(N, 0, -1):
        for a in range(3):
            suffixX[a][i] = suffixX[a][i+1]
        if S[i-1] == 'X':
            suffixX[A[i-1]][i] += 1

    ans = 0

    # For each j in [1..N], if S_j = 'E', then we want all i in Mpos with i<j,
    # and k in Xpos with k>j. So we sum over a in {0,1,2} and c in {0,1,2}:
    #    mex(a, b, c) * (#M with A= a up to j-1) * (#X with A= c after j)
    # where b = A[j-1].
    for j in range(1, N+1):
        if S[j-1] == 'E':
            b = A[j-1]
            for a in range(3):
                countM = prefixM[a][j-1]
                if countM == 0:
                    continue
                for c in range(3):
                    countX = suffixX[c][j+1]  # positions strictly after j
                    if countX != 0:
                        ans += mex_table[a][b][c] * countM * countX

    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()