# YOUR CODE HERE

import sys

def solve(N, S):
    count = 0
    for i in range(N):
        for j in range(i+1, N+1):
            if S[i:j] == S[0]*len(S[i:j]):
                count += 1
    return count

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

print(solve(N, S))