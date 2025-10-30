n = int(input())
users = []
for _ in range(n):
    s, c = input().split()
    users.append((s, int(c)))

# Sort the users lexicographically by their username
sorted_users = sorted(users, key=lambda x: x[0])

# Calculate the total sum of ratings
total_rating = sum(c for s, c in users)

# Determine the index of the winner
winner_index = total_rating % n

# Output the username of the winner
print(sorted_users[winner_index][0])