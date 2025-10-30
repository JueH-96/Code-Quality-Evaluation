import sys
from itertools import permutations

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
conditions = []
for i in range(M):
    L = int(data[2 + 3 * i]) - 1
    R = int(data[3 + 3 * i]) - 1
    X = int(data[4 + 3 * i]) - 1
    conditions.append((L, R, X))

MOD = 998244353

def is_valid_permutation(P):
    for L, R, X in conditions:
        if max(P[L:R+1]) == P[X]:
            return False
    return True

count = 0
for P in permutations(range(N)):
    if is_valid_permutation(P):
        count += 1
        count %= MOD

print(count)