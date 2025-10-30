# YOUR CODE HERE
import sys

# Read input
N = int(input())
users = []
for _ in range(N):
    S, C = input().split()
    C = int(C)
    users.append((S, C))

# Sort users by their usernames in lexicographical order
users.sort(key=lambda x: x[0])

# Calculate the sum of the ratings
total_rating = sum(C for _, C in users)

# Determine the winner
winner_index = total_rating % N
winner_name = users[winner_index][0]

# Print the winner's username
print(winner_name)