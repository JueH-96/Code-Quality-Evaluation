import sys

def main():
    N = int(sys.stdin.readline())
    users_data = []
    total_rating = 0
    for _ in range(N):
        line = sys.stdin.readline().split()
        username = line[0]
        rating = int(line[1])
        users_data.append((username, rating))
        total_rating += rating

    # Sort users by username lexicographically.
    # Python's default sort for tuples sorts by the first element, then the second, etc.
    # Sorting a list of (username, rating) tuples naturally sorts by username first.
    users_data.sort()

    # Calculate the index of the winner in the sorted list.
    # The user assigned number k is the user at index k in the sorted list.
    # The winning number is total_rating % N.
    winner_index = total_rating % N

    # Get the username of the user at the winner_index in the sorted list.
    winner_username = users_data[winner_index][0]

    print(winner_username)

if __name__ == "__main__":
    main()