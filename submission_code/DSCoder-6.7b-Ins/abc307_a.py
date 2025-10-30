# YOUR CODE HERE

N = int(input())
steps = list(map(int, input().split()))

total_steps = []
for i in range(N):
    total_steps.append(sum(steps[i*7:(i+1)*7]))

print(' '.join(map(str, total_steps)))