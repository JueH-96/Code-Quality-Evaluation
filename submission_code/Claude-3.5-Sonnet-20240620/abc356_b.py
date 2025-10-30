# YOUR CODE HERE
N, M = map(int, input().split())
goals = list(map(int, input().split()))
total_nutrients = [0] * M

for _ in range(N):
    food = list(map(int, input().split()))
    for j in range(M):
        total_nutrients[j] += food[j]

if all(total_nutrients[j] >= goals[j] for j in range(M)):
    print("Yes")
else:
    print("No")