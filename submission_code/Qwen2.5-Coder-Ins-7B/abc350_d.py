# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Create a list of sets to represent the graph
graph = [set() for _ in range(N + 1)]

# Add edges to the graph
for i in range(M):
    A = int(data[2 + 2 * i])
    B = int(data[3 + 2 * i])
    graph[A].add(B)
    graph[B].add(A)

# Initialize the count of new friendships
new_friendships = 0

# Iterate over all pairs of users
for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        # Check if users i and j are friends
        if j in graph[i]:
            # Check if there is a common friend between i and j
            for k in range(j + 1, N + 1):
                if k in graph[i] and k in graph[j]:
                    # If there is a common friend, increment the count of new friendships
                    new_friendships += 1

# Print the result
print(new_friendships)