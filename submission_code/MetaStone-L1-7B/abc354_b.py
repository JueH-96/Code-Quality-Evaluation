n = int(input())
users = []
for _ in range(n):
    s, c = input().split()
    users.append((s, int(c)))

sorted_users = sorted(users, key=lambda x: x[0])
total = sum(c for s, c in sorted_users)
target = total % n

print(sorted_users[target][0])