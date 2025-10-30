def maximum_new_friendships(N, M, friendships):
    from collections import defaultdict

    # Create an adjacency list for the friendships
    friends = defaultdict(set)
    for A, B in friendships:
        friends[A].add(B)
        friends[B].add(A)

    count = 0

    # For each user, check their friends
    for Y in range(1, N + 1):
        # Get friends of Y
        friends_of_Y = list(friends[Y])
        num_friends = len(friends_of_Y)

        # Check pairs of friends of Y
        for i in range(num_friends):
            for j in range(i + 1, num_friends):
                X = friends_of_Y[i]
                Z = friends_of_Y[j]
                # Check if X and Z are not friends
                if Z not in friends[X]:
                    count += 1

    return count

import sys
input = sys.stdin.read
data = input().splitlines()

# Read N and M
N, M = map(int, data[0].split())
friendships = [tuple(map(int, line.split())) for line in data[1:M + 1]]

# Get the result
result = maximum_new_friendships(N, M, friendships)

# Print the result
print(result)