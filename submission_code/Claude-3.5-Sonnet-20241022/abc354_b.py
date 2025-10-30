N = int(input())
users = []
total_rating = 0

# Read input
for _ in range(N):
    name, rating = input().split()
    rating = int(rating)
    users.append((name, rating))
    total_rating += rating

# Sort users by username lexicographically
sorted_users = sorted(users, key=lambda x: x[0])

# Find winner index using modulo
winner_index = total_rating % N

# Print winner's name
print(sorted_users[winner_index][0])