N = int(input())
users = []

for _ in range(N):
    line = input().strip()
    S, C = line.split()
    C = int(C)
    users.append((S, C))

# Sort users lexicographically by username
sorted_users = sorted(users, key=lambda x: x[0])

# Calculate the sum of ratings
T = sum(user[1] for user in sorted_users)

# Determine the winner's index
winner_number = T % N

# Print the winner's username
print(sorted_users[winner_number][0])