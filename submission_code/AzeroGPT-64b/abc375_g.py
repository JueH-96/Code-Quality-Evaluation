from heapq import heappop, heappush

def findShortestPathDifference(N, M, roads):
    # 1. Initialize the graph and basic arrays
    graph = [[] for _ in range(N + 1)]
    for a, b, c in roads:
        graph[a].append((b, c))
        graph[b].append((a, c))
    distances = [float('inf')] * (N + 1)
    distances_roadExcluded = [float('inf')] * (N + 1)
    distances[1] = 0

    # 2. Dijkstra's algorithm for whole graph from city 1 to city N
    priority_queue = [(0, 1)]
    while priority_queue:
        current_distance, current_city = heappop(priority_queue)
        if current_distance > distances[current_city]:
            continue
        for neighbor, weight in graph[current_city]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(priority_queue, (distance, neighbor))

    # 3. Modify distances for each road excluded
    differences = []
    for idx, road in enumerate(roads):
        a, b, _ = road

        # Reset distances for each case
        distances_roadExcluded = distances[:]

        # Delete the weights of the excluded road
        for x in [a, b]:
            for i, (n, w) in enumerate(graph[x]):
                if n in [a, b]:
                    graph[x][i] = (n, float('inf'))

        # Dijkstra's algorithm for when road i is excluded
        priority_queue = [(0, 1)]
        while priority_queue:
            current_distance, current_city = heappop(priority_queue)
            if current_distance > distances_roadExcluded[current_city]:
                continue
            for neighbor, weight in graph[current_city]:
                distance = current_distance + weight
                if distance < distances_roadExcluded[neighbor]:
                    distances_roadExcluded[neighbor] = distance
                    heappush(priority_queue, (distance, neighbor))

        # Reset weights of the excluded road
        for x in [a, b]:
            for i, (n, w) in enumerate(graph[x]):
                if n in [a, b]:
                    graph[x][i] = (n, roads[idx][2])

        # Check if distance is different or one is unreachable
        answer = False
        if (distances[1] == float('inf') or distances[N] == float('inf')) and 0 not in (distances_roadExcluded[1], distances_roadExcluded[N]):
            answer = True
        elif distances_roadExcluded[1] == float('inf') or distances_roadExcluded[N] == float('inf') or distances[N] != distances_roadExcluded[N]:
            answer = True
        differences.append('Yes' if answer else 'No')

    return differences


# Read input
N, M = map(int, input().split())
roads = [tuple(map(int, input().split())) for _ in range(M)]

# Solve and write output
differences = findShortestPathDifference(N, M, roads)
for diff in differences:
    print(diff)