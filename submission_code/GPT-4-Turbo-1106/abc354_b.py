# Read the number of users
N = int(input())

# Initialize a list to store the usernames and ratings
users = []

# Read the usernames and ratings
for _ in range(N):
    username, rating = input().split()
    rating = int(rating)
    users.append((username, rating))

# Sort the users by their usernames in lexicographical order
users.sort(key=lambda x: x[0])

# Calculate the sum of the ratings
total_rating = sum(rating for _, rating in users)

# Find the winner's index
winner_index = total_rating % N

# Print the winner's username
print(users[winner_index][0])