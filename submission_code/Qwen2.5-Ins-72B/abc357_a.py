# YOUR CODE HERE
N, M = map(int, input().split())
H = list(map(int, input().split()))

count = 0
for i in range(N):
    if M >= H[i]:
        M -= H[i]
        count += 1
    else:
        break

print(count)