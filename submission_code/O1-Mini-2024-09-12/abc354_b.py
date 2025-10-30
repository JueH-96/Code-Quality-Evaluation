n = int(input())
users = []
total = 0
for _ in range(n):
    s, c = input().split()
    c = int(c)
    users.append((s, c))
    total += c

users_sorted = sorted(users, key=lambda x: x[0])
winner_index = total % n
print(users_sorted[winner_index][0])