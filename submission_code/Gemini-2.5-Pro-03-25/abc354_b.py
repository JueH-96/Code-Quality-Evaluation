# YOUR CODE HERE
import sys

def solve():
    # Read the number of users from standard input
    n = int(sys.stdin.readline())

    # Initialize a list to store user data. Each element will be a tuple (username, rating).
    users = []
    # Initialize a variable to store the sum of all ratings.
    total_rating = 0

    # Loop N times to read the data for each user
    for _ in range(n):
        # Read the line containing username and rating
        line = sys.stdin.readline().split()
        # Extract username (string)
        s = line[0]
        # Extract rating (string) and convert it to an integer
        c_str = line[1]
        c = int(c_str)
        
        # Store the user data as a tuple in the list
        users.append((s, c))
        # Add the current user's rating to the total sum
        total_rating += c

    # Sort the list of users. Python's default sort for a list of tuples
    # sorts based on the first element of the tuple (the username in this case).
    # This automatically handles lexicographical sorting as required.
    users.sort()

    # Calculate the index of the winner according to the game rules:
    # The winner is the user assigned the number T mod N, where T is the total rating.
    # Since the list 'users' is now sorted lexicographically, the user assigned number i
    # corresponds to the element at index i in the list.
    winner_index = total_rating % n

    # Retrieve the username of the winner. The winner is the user at the calculated
    # index in the sorted list. The username is the first element of the tuple.
    winner_username = users[winner_index][0]

    # Print the winner's username to standard output
    print(winner_username)

# Call the solve function to run the program
solve()