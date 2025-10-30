# YOUR CODE HERE

N, M = map(int, input().split())
S = input()
C = list(map(int, input().split()))

for i in range(1, M + 1):
    shift = C[i - 1]
    S = S[shift:] + S[:shift]

print(S)