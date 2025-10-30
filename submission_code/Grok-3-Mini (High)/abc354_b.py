import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Read N from the first element
N = int(data[0])

# Extract usernames and ratings using slicing
usernames = data[1::2]  # List of usernames in input order
ratings_str = data[2::2]  # List of rating strings in input order

# Convert ratings to integers
ratings = [int(r) for r in ratings_str]

# Calculate the sum of all ratings
sum_T = sum(ratings)

# Sort the usernames in lexicographical order
sorted_usernames = sorted(usernames)

# Compute the winner's index using sum_T modulo N
winner_index = sum_T % N

# Get the winner's username from the sorted list
winner_name = sorted_usernames[winner_index]

# Print the winner's username
print(winner_name)