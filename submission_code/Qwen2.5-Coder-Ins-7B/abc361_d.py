# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]
T = data[2]

def min_operations(N, S, T):
    operations = 0
    for i in range(N):
        if S[i] != T[i]:
            j = i + 1
            while j < N and S[j] == T[j]:
                j += 1
            if j == N:
                return -1
            operations += j - i
            S = S[:i] + S[j] + S[i+1:j] + S[i] + S[j+1:]
    return operations

print(min_operations(N, S, T))