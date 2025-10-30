def min_operations_to_get_1_and_M(N, M, sets):
    from collections import defaultdict, deque

    # Create a graph where each set is a node and edges exist between sets that share elements
    graph = defaultdict(set)
    element_to_sets = defaultdict(set)

    for index, s in enumerate(sets):
        for element in s:
            element_to_sets[element].add(index)

    for indices in element_to_sets.values():
        indices = list(indices)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                graph[indices[i]].add(indices[j])
                graph[indices[j]].add(indices[i])

    # BFS to find the minimum number of operations to connect sets containing 1 and M
    start_sets = set(element_to_sets[1])
    target_sets = set(element_to_sets[M])

    if not start_sets or not target_sets:
        return -1

    queue = deque(start_sets)
    visited = set(start_sets)
    operations = 0

    while queue:
        for _ in range(len(queue)):
            current_set = queue.popleft()
            if current_set in target_sets:
                return operations

            for neighbor in graph[current_set]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        operations += 1

    return -1

import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
sets = []
index = 1

for _ in range(N):
    A_i = int(data[index])
    index += 1
    S_i = list(map(int, data[index].split()))
    sets.append(S_i)
    index += 1

result = min_operations_to_get_1_and_M(N, M, sets)
print(result)