import sys

def solve():
    N = int(sys.stdin.readline())

    users_info = []
    total_rating_sum = 0

    for _ in range(N):
        s, c_str = sys.stdin.readline().split()
        c = int(c_str)
        users_info.append((s, c))
        total_rating_sum += c

    # Extract only usernames to sort them lexicographically
    # The assigned numbers 0, 1, ..., N-1 correspond to the index
    # in this sorted list of usernames.
    usernames = [info[0] for info in users_info]
    
    # Sort the usernames lexicographically
    # Python's default string sorting is lexicographical
    sorted_usernames = sorted(usernames)

    # Calculate the assigned number of the winner
    # This is (sum of all ratings) % N
    winner_assigned_number = total_rating_sum % N

    # The winner's username is at the calculated index in the sorted list
    winner_username = sorted_usernames[winner_assigned_number]

    sys.stdout.write(winner_username + "
")

if __name__ == '__main__':
    solve()