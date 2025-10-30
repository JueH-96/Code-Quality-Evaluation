# Read the number of users
N = int(input())

# Read the usernames and ratings
users = []
for _ in range(N):
    S, C = input().split()
    users.append((S, int(C)))

# Sort the users lexicographically by username
sorted_users = sorted(users, key=lambda x: x[0])

# Calculate the total sum of ratings
total = sum(c for _, c in users)

# Determine the winner's index
winner_index = total % N

# Get the winner's username
winner = sorted_users[winner_index][0]

# Print the winner
print(winner)