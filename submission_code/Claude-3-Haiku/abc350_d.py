from collections import defaultdict

def max_new_friendships(N, M, friendships):
    # Create a graph to represent the friendships
    graph = defaultdict(list)
    for a, b in friendships:
        graph[a].append(b)
        graph[b].append(a)
    
    # Count the number of new friendships that can be formed
    count = 0
    for x in range(1, N+1):
        for y in graph[x]:
            for z in graph[y]:
                if x != z and z not in graph[x]:
                    count += 1
    
    return count

# Read input from stdin
N, M = map(int, input().split())
friendships = []
for _ in range(M):
    a, b = map(int, input().split())
    friendships.append((a, b))

# Solve the problem and print the answer
print(max_new_friendships(N, M, friendships))