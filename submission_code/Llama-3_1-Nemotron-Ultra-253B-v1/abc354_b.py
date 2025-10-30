n = int(input())
users = []
for _ in range(n):
    s, c = input().split()
    users.append((s, int(c)))
users.sort(key=lambda x: x[0])
total = sum(c for s, c in users)
index = total % n
print(users[index][0])