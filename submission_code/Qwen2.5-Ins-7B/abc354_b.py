# YOUR CODE HERE
n = int(input())
users = []
for i in range(n):
    s, c = input().split()
    users.append((s, int(c)))
total_rating = sum(c for _, c in users)
winner_index = total_rating % n
winner = min(users, key=lambda x: (x[0], -x[1]))
print(winner[0])