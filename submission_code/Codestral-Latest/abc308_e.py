# YOUR CODE HERE
import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
S = data[N+1]

# Precompute the mex values for all triples (i, j, k)
mex_values = {}
for i, j, k in combinations(range(N), 3):
    if S[i] == 'M' and S[j] == 'E' and S[k] == 'X':
        values = {A[i], A[j], A[k]}
        if 0 not in values:
            mex_values[(i, j, k)] = 0
        elif 1 not in values:
            mex_values[(i, j, k)] = 1
        else:
            mex_values[(i, j, k)] = 2

# Sum the mex values
total_mex = sum(mex_values.values())

print(total_mex)