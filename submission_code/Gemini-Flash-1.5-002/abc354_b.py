# YOUR CODE HERE
n = int(input())
users = []
for i in range(n):
    s, c = input().split()
    users.append((s, int(c)))

users.sort()
total_rating = sum([c for s, c in users])
winner_index = total_rating % n
print(users[winner_index][0])