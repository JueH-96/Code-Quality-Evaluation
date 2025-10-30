# Read the input
N = int(input())
users = []
for _ in range(N):
    s, c = input().split()
    users.append((s, int(c)))

# Sort the users in lexicographical order
users.sort(key=lambda x: x[0])

# Calculate the sum of the ratings
total_rating = sum(c for _, c in users)

# Find the winner
winner_index = total_rating % N
winner_name = users[winner_index][0]

# Print the winner's username
print(winner_name)