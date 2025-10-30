# Read N, the number of users
N = int(input())

# Initialize a list to store user data as (name, rating) tuples
users_data = []
# Initialize total_rating to sum up all ratings
total_rating = 0

# Loop N times to read each user's data
for _ in range(N):
    # Read the line containing name and rating, split by space
    name_str, rating_str = input().split()
    # Convert rating string to integer
    rating_int = int(rating_str)
    
    # Add the (name, rating) tuple to our list
    users_data.append((name_str, rating_int))
    # Add the current user's rating to the total sum
    total_rating += rating_int

# Sort the users_data list.
# Python's default sort for a list of tuples sorts them based on the first element 
# (in this case, name_str). If first elements are identical (which they are not, as per problem
# statement that names are distinct), it would then compare second elements (rating_int).
# So, this correctly sorts users lexicographically by their names.
users_data.sort()

# After sorting, users_data[0] is the user assigned number 0,
# users_data[1] is the user assigned number 1, and so on, up to users_data[N-1] for number N-1.

# Calculate the winning number according to the rule: T mod N.
# T is the sum of all ratings (total_rating).
winning_assignment_number = total_rating % N

# The winner is the user who was assigned winning_assignment_number.
# This corresponds to the user at index winning_assignment_number in our sorted list.
# The winner's name is the first element (index 0) of the tuple at this list index.
winner_name = users_data[winning_assignment_number][0]

# Print the winner's name to standard output.
print(winner_name)