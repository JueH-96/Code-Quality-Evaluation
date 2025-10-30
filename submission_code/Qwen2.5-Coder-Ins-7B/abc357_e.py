# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
a = list(map(int, data[1:]))

# Create a list to store the length of the path to each node
path_length = [0] * (N + 1)

# Traverse the graph to calculate the path length for each node
current_node = 1
while current_node != -1:
    path_length[current_node] = 1
    current_node = a[current_node - 1]

# Calculate the number of reachable pairs
reachable_pairs = 0
for i in range(1, N + 1):
    reachable_pairs += path_length[i] * (N - path_length[i] + 1)

print(reachable_pairs)