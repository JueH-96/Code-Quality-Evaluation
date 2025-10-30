# YOUR CODE HERE

N = int(input())
sticks = [input() for _ in range(N)]

count = 0
for i in range(N):
    for j in range(i+1, N):
        if sticks[i] == sticks[j][::-1]:
            count += 1
            break

print(count)