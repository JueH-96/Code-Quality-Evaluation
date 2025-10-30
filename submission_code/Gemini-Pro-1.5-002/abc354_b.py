# YOUR CODE HERE
n = int(input())
users = []
total_rating = 0
for _ in range(n):
    name, rating = input().split()
    users.append((name, int(rating)))
    total_rating += int(rating)

users.sort()

winner_index = total_rating % n
winner_name = users[winner_index][0]

print(winner_name)