n = int(input())
users = []
for _ in range(n):
    s, c = input().split()
    c = int(c)
    users.append((s, c))

# Sort the users based on their usernames lexicographically
sorted_users = sorted(users, key=lambda x: x[0])

# Calculate the total sum of ratings
total = sum(c for s, c in sorted_users)

# Determine the winner's index
index = total % n

# Output the winner's username
print(sorted_users[index][0])