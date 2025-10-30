# Read the number of users
N = int(input())

# Read the usernames and ratings
users = []
for _ in range(N):
    username, rating = input().split()
    users.append((username, int(rating)))

# Sort the users in lexicographical order
users.sort(key=lambda x: x[0])

# Calculate the sum of ratings
total_rating = sum(user[1] for user in users)

# Calculate the winner's index
winner_index = total_rating % N

# Print the winner's username
print(users[winner_index][0])