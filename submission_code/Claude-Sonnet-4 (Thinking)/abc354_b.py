n = int(input())
users = []
total_rating = 0

for _ in range(n):
    parts = input().split()
    name = parts[0]
    rating = int(parts[1])
    users.append(name)
    total_rating += rating

# Sort names lexicographically
users.sort()

# Find winner
winner_index = total_rating % n
print(users[winner_index])