# YOUR CODE HERE
N, D = map(int, input().split())
S = [input() for _ in range(N)]
days_free = [True]*D
for i in range(N):
    for d in range(D):
        if S[i][d] == 'x':
            days_free[d] = False

max_consecutive = 0
current_consecutive = 0
for d in range(D):
    if days_free[d]:
        current_consecutive +=1
        max_consecutive = max(max_consecutive, current_consecutive)
    else:
        current_consecutive = 0

print(max_consecutive)