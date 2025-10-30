import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Create a graph to represent friendships
graph = defaultdict(set)

# Read the friendships and add them to the graph
for i in range(2, 2 + 2 * M, 2):
    A = int(data[i]) - 1
    B = int(data[i + 1]) - 1
    graph[A].add(B)
    graph[B].add(A)

# Function to perform the operation and count the number of new friendships
def count_new_friendships():
    count = 0
    for X in range(N):
        for Y in graph[X]:
            for Z in graph[Y]:
                if Z != X and Z not in graph[X]:
                    count += 1
    return count // 6  # Each new friendship is counted 6 times

# Calculate the maximum number of operations
result = count_new_friendships()
print(result)