import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = data[N+1]

    # Lists to store counts of 'M', 'E', 'X' at each index
    count_M = [0] * (N + 1)
    count_E = [0] * (N + 1)
    count_X = [0] * (N + 1)

    # Count occurrences of 'M', 'E', 'X'
    for i in range(N):
        if S[i] == 'M':
            count_M[i+1] = count_M[i] + 1
            count_E[i+1] = count_E[i]
            count_X[i+1] = count_X[i]
        elif S[i] == 'E':
            count_M[i+1] = count_M[i]
            count_E[i+1] = count_E[i] + 1
            count_X[i+1] = count_X[i]
        else:
            count_M[i+1] = count_M[i]
            count_E[i+1] = count_E[i]
            count_X[i+1] = count_X[i] + 1

    # Lists to store counts of (0, 1, 2) in A up to each index
    count_0 = [0] * (N + 1)
    count_1 = [0] * (N + 1)
    count_2 = [0] * (N + 1)

    # Count occurrences of 0, 1, 2 in A
    for i in range(N):
        if A[i] == 0:
            count_0[i+1] = count_0[i] + 1
            count_1[i+1] = count_1[i]
            count_2[i+1] = count_2[i]
        elif A[i] == 1:
            count_0[i+1] = count_0[i]
            count_1[i+1] = count_1[i] + 1
            count_2[i+1] = count_2[i]
        else:
            count_0[i+1] = count_0[i]
            count_1[i+1] = count_1[i]
            count_2[i+1] = count_2[i] + 1

    total_sum = 0

    # Calculate the sum of mex(A_i, A_j, A_k) for all valid (i, j, k)
    for i in range(N):
        if S[i] == 'M':
            mex_0 = count_E[i] * (count_X[N] - count_X[i]) - count_1[i] * (count_2[N] - count_2[i])
            mex_1 = count_E[i] * (count_X[N] - count_X[i]) - (count_E[i] - count_1[i]) * (count_X[N] - count_X[i]) + count_0[i] * (count_2[N] - count_2[i])
            mex_2 = (count_E[N] - count_E[i]) * count_X[i] - (count_1[N] - count_1[i]) * count_2[i]
            total_sum += mex_0 + mex_1 + mex_2

    print(total_sum)

if __name__ == "__main__":
    solve()