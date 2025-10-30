# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    idx = 1
    users = []
    for _ in range(N):
        name = data[idx]
        rating = int(data[idx + 1])
        users.append((name, rating))
        idx += 2

    # Sum all ratings
    total_rating = sum(r for _, r in users)

    # Sort by username in lexicographical order
    users.sort(key=lambda x: x[0])

    # Determine the winner's index
    winner_index = total_rating % N

    # Print the corresponding username
    print(users[winner_index][0])

# Do not forget to call main()
main()