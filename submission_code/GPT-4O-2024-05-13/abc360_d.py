# YOUR CODE HERE
def count_passing_pairs(N, T, S, X):
    pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            if S[i] == '1' and S[j] == '0' and X[i] < X[j]:
                if (X[j] - X[i]) <= 2 * T:
                    pairs += 1
    return pairs

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
S = data[2]
X = list(map(int, data[3:]))

print(count_passing_pairs(N, T, S, X))