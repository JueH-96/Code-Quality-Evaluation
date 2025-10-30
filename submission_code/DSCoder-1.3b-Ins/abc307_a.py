# YOUR CODE HERE

N = int(input().split()[0])
steps = list(map(int, input().split()))

B = [0]*N

for i in range(N):
    B[i] = sum(steps[i:i+7])

print(*B)