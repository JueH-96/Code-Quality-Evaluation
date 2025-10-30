import sys

def solve():
    N, M = map(int, input().split())
    X, Y, Z = [], [], []
    for _ in range(M):
        x, y, z = map(int, input().split())
        X.append(x)
        Y.append(y)
        Z.append(z)

    # Create a dictionary to store the required XOR values for each position
    xor_dict = {}
    for i in range(M):
        xor_dict[(X[i], Y[i])] = Z[i]

    # Find the minimum sum sequence
    A = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            if (i+1, j+1) in xor_dict and A[i] ^ A[j] != xor_dict[(i+1, j+1)]:
                A[i] = xor_dict[(i+1, j+1)] ^ A[j]
                break
        else:
            return -1

    return ' '.join(map(str, A))

print(solve())