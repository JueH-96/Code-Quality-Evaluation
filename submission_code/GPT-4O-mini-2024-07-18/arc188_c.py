def find_confused_villagers(N, M, testimonies):
    from collections import defaultdict, deque

    # Graph to store the testimonies
    graph = defaultdict(list)
    
    for A, B, C in testimonies:
        # C=0 means A says B is honest (A -> B is honest)
        # C=1 means A says B is a liar (A -> B is a liar)
        if C == 0:
            graph[A].append((B, 0))  # A says B is honest
            graph[B].append((A, 0))  # B is honest means A is honest
        else:
            graph[A].append((B, 1))  # A says B is a liar
            graph[B].append((A, 1))  # B is a liar means A is a liar

    # To keep track of the color (0 or 1) assigned to each villager
    color = [-1] * (N + 1)  # -1 means unvisited

    def bfs(start):
        queue = deque([start])
        color[start] = 0  # Start coloring with 0
        while queue:
            node = queue.popleft()
            for neighbor, relation in graph[node]:
                expected_color = color[node] if relation == 0 else 1 - color[node]
                if color[neighbor] == -1:  # Not colored yet
                    color[neighbor] = expected_color
                    queue.append(neighbor)
                elif color[neighbor] != expected_color:  # Conflict found
                    return False
        return True

    # Check for each component of the graph
    for villager in range(1, N + 1):
        if color[villager] == -1:  # Not visited yet
            if not bfs(villager):
                return "-1"

    # Construct the output string
    confused = ['0'] * N
    for i in range(1, N + 1):
        if color[i] == 1:  # If colored 1, mark as confused
            confused[i - 1] = '1'

    return ''.join(confused)

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
testimonies = [tuple(map(int, line.split())) for line in data[1:M + 1]]

# Get the result
result = find_confused_villagers(N, M, testimonies)

# Print the result
print(result)