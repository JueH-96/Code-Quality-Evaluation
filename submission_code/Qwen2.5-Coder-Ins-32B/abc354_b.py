# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
users = []

for i in range(N):
    name = input[2 * i + 1]
    rating = int(input[2 * i + 2])
    users.append((name, rating))

users.sort(key=lambda x: x[0])
total_rating = sum(user[1] for user in users)
winner_index = total_rating % N
print(users[winner_index][0])