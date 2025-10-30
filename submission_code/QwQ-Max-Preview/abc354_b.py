n = int(input())
users = []
for _ in range(n):
    s, c = input().split()
    users.append((s, int(c)))

# Sort the users lexicographically by their username
sorted_users = sorted(users, key=lambda x: x[0])

# Calculate the total rating
total_rating = sum(c for _, c in users)

# Determine the winner's index
index = total_rating % n

# Output the corresponding username
print(sorted_users[index][0])