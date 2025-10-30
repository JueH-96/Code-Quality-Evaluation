# YOUR CODE HERE
n = int(input())
users = []

for _ in range(n):
    line = input().split()
    name = line[0]
    rating = int(line[1])
    users.append((name, rating))

# Sort users by name in lexicographical order
users.sort(key=lambda x: x[0])

# Calculate sum of all ratings
total_rating = sum(rating for name, rating in users)

# Find winner position
winner_position = total_rating % n

# Print winner's name
print(users[winner_position][0])