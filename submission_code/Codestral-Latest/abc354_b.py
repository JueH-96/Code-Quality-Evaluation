# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
users = []

for i in range(N):
    name = data[2 * i + 1]
    rating = int(data[2 * i + 2])
    users.append((name, rating))

# Sort users by lexicographical order of their usernames
users.sort()

# Calculate the sum of the ratings
total_rating = sum(rating for _, rating in users)

# Find the winner's index
winner_index = total_rating % N

# The winner's username
winner_name = users[winner_index][0]

print(winner_name)