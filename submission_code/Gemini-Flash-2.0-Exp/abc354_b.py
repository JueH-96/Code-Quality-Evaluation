n = int(input())
users = []
total_rating = 0
for _ in range(n):
    s, c = input().split()
    users.append((s, int(c)))
    total_rating += int(c)

users.sort(key=lambda x: x[0])
winner_index = total_rating % n
print(users[winner_index][0])