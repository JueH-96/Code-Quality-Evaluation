# YOUR CODE HERE

N, T = map(int, input().split())
S = input()
X = list(map(int, input().split()))

ants = sorted([(X[i], S[i]) for i in range(N)], key=lambda x: x[0])

count = 0
for i in range(N - 1):
    if ants[i][1] != ants[i + 1][1]:
        count += 1

print(count)