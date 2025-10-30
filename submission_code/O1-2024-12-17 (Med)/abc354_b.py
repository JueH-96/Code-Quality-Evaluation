def main():
    import sys

    N = int(sys.stdin.readline().strip())
    users = []
    total = 0

    for _ in range(N):
        s, c = sys.stdin.readline().split()
        c = int(c)
        users.append((s, c))
        total += c

    # Sort by username (lexicographical order)
    users.sort(key=lambda x: x[0])

    # Calculate the winner's index
    winner_index = total % N

    # Print the winner's username
    print(users[winner_index][0])


# Do not remove the call to main()
main()