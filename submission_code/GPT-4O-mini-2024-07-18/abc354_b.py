# YOUR CODE HERE
N = int(input())
users = []

for _ in range(N):
    line = input().split()
    username = line[0]
    rating = int(line[1])
    users.append((username, rating))

# Sort users by username lexicographically
users.sort()

# Calculate the total rating
total_rating = sum(rating for _, rating in users)

# Determine the winner's index
winner_index = total_rating % N

# Print the winner's username
print(users[winner_index][0])