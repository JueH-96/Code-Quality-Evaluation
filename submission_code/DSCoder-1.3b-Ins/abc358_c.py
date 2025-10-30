# YOUR CODE HERE

N, M = map(int, input().split())
stands = [input() for _ in range(N)]

min_stands = 0
for i in range(N):
    if 'o' in stands[i]:
        min_stands += 1

print(min_stands)