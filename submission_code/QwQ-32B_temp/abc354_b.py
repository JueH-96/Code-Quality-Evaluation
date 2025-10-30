n = int(input())
users = []
for _ in range(n):
    s, c = input().split()
    users.append((s, int(c)))

total_c = sum(c for s, c in users)
sorted_users = sorted(users, key=lambda x: x[0])
index = total_c % n
print(sorted_users[index][0])