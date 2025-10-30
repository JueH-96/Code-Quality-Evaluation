def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:1+N]))
    S = input_data[1+N]

    # We only care about positions where S[i]=='M', S[i]=='E', or S[i]=='X'.

    # prefixM[val][j] = number of indices i < j (1-based) with S_i='M' and A_i=val
    # We'll use 0-based internally but shift carefully
    prefixM = [[0]*(N+1) for _ in range(3)]
    for i in range(N):
        for val in range(3):
            prefixM[val][i+1] = prefixM[val][i]
        if S[i] == 'M':
            prefixM[A[i]][i+1] += 1

    # suffixX[val][j] = number of indices k >= j (1-based) with S_k='X' and A_k=val
    # We'll fill from the right
    suffixX = [[0]*(N+2) for _ in range(3)]
    for i in range(N-1, -1, -1):
        for val in range(3):
            suffixX[val][i+1] = suffixX[val][i+2]
        if S[i] == 'X':
            suffixX[A[i]][i+1] += 1

    def mex(a, b, c):
        vals = {a, b, c}
        if 0 not in vals:
            return 0
        if 1 not in vals:
            return 1
        if 2 not in vals:
            return 2
        return 3

    ans = 0
    # For each j where S_j='E', we combine M's to the left and X's to the right
    for j in range(N):
        if S[j] == 'E':
            aj = A[j]
            # count of M with val_i = prefixM[val_i][j] (since i < j => i <= j-1, j is 0-based)
            # count of X with val_k = suffixX[val_k][j+2] (since k>j => k >= j+1 => in 1-based indexing that is index j+2 in suffix array)
            left_counts = [prefixM[val][j] for val in range(3)]
            right_counts = [suffixX[val][j+2] for val in range(3)]
            for val_i in range(3):
                for val_k in range(3):
                    cnt_i = left_counts[val_i]
                    cnt_k = right_counts[val_k]
                    if cnt_i > 0 and cnt_k > 0:
                        ans += cnt_i * cnt_k * mex(val_i, aj, val_k)

    print(ans)

# Call main() so that the solution is executed
main()