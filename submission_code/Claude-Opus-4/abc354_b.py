# YOUR CODE HERE
n = int(input())
users = []

# Read all users
for _ in range(n):
    line = input().split()
    name = line[0]
    rating = int(line[1])
    users.append((name, rating))

# Sort users by name in lexicographical order
users.sort(key=lambda x: x[0])

# Calculate sum of ratings
total_rating = sum(rating for name, rating in users)

# Find winner index
winner_index = total_rating % n

# Print winner's name
print(users[winner_index][0])