# YOUR CODE HERE
N, M = map(int, input().split())
H = list(map(int, input().split()))

count = 0
remaining = M

for i in range(N):
    if remaining >= H[i]:
        remaining -= H[i]
        count += 1
    else:
        break

print(count)