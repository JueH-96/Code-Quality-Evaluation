from collections import defaultdict

def find_max_operations(N, M, friendships):
    graph = defaultdict(set)
    for a, b in friendships:
        graph[a].add(b)
        graph[b].add(a)
    
    operations = 0
    for node in range(1, N + 1):
        for friend in list(graph[node]):
            for friends_friend in list(graph[friend]):
                if friends_friend != node and node not in graph[friends_friend]:
                    graph[node].add(friends_friend)
                    graph[friends_friend].add(node)
                    operations += 1
    return operations

# Read input from stdin
N, M = map(int, input().split())
friendships = [tuple(map(int, input().split())) for _ in range(M)]

# Solve the problem
max_operations = find_max_operations(N, M, friendships)

# Write the answer to stdout
print(max_operations)