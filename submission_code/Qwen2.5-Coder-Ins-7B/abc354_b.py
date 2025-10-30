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

users.sort()

total_rating = sum(rating for _, rating in users)
winner_index = total_rating % N

winner_name = users[winner_index][0]
print(winner_name)