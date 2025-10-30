# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

boxes = []
for i in range(N):
    boxes.append((A[i], A[i]))

people = []
for i in range(M):
    people.append(B[i])

boxes.sort()
people.sort()

ans = 0
satisfied = [False] * M
count = 0

for i in range(N):
    for j in range(M):
        if not satisfied[j] and boxes[i][1] >= people[j]:
            ans += boxes[i][0]
            satisfied[j] = True
            count += 1
            break
    if count == M:
        break

if count == M:
    print(ans)
else:
    print(-1)