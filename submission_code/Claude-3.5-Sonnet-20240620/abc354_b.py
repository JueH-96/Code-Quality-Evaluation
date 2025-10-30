# YOUR CODE HERE
N = int(input())
users = []
total_rating = 0

for _ in range(N):
    name, rating = input().split()
    users.append((name, int(rating)))
    total_rating += int(rating)

users.sort(key=lambda x: x[0])  # Sort users by name lexicographically
winner_index = total_rating % N

print(users[winner_index][0])