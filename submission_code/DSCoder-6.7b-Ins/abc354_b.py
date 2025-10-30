# YOUR CODE HERE

N = int(input())
users = []
for _ in range(N):
    S, C = input().split()
    users.append((S, int(C)))

users.sort()

total_rating = sum(c for _, c in users)
winner_index = total_rating % N

print(users[winner_index][0])