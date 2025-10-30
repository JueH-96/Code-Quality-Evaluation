# YOUR CODE HERE
M = int(input())
S = [input() for _ in range(3)]

for t in range(M + 1):
    if S[0][t % M] == S[1][t % M] == S[2][t % M]:
        print(t)
        break
else:
    print(-1)