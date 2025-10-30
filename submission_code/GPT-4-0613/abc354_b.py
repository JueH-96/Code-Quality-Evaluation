N = int(input())
users = []
for _ in range(N):
    S, C = input().split()
    C = int(C)
    users.append((S, C))

users.sort()
T = sum(C for _, C in users) % N
print(users[T][0])