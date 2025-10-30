# YOUR CODE HERE
N = int(input())
users = []
total_rating = 0
for _ in range(N):
    S_i, C_i = input().split()
    C_i = int(C_i)
    users.append((S_i, C_i))
    total_rating += C_i

# Sort the usernames lexicographically
users_sorted = sorted(users, key=lambda x: x[0])

# Map assigned number to username
assigned_numbers = {}
for idx, (S_i, _) in enumerate(users_sorted):
    assigned_numbers[idx] = S_i

winner_number = total_rating % N
print(assigned_numbers[winner_number])