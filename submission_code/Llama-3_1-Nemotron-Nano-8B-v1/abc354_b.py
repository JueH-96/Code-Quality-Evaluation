n = int(input())
users = []
for _ in range(n):
    s, c = input().split()
    users.append((s, int(c)))

users_sorted = sorted(users, key=lambda x: x[0])
total = sum(c for s, c in users_sorted)
winner_index = total % n
print(users_sorted[winner_index][0])