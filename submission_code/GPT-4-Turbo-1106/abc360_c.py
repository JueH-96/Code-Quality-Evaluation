import sys
from collections import defaultdict

def find_min_cost(N, box_to_items, weights):
    # Create a graph where each node represents a box, and each edge represents a possible move
    graph = defaultdict(list)
    for item, boxes in box_to_items.items():
        for i in range(len(boxes) - 1):
            graph[boxes[i]].append((boxes[i + 1], weights[item - 1]))

    # Perform a BFS to find the minimum cost to move items to their correct boxes
    visited = set()
    min_cost = 0
    for start in range(1, N + 1):
        if start in visited:
            continue
        queue = [(start, 0)]
        local_visited = set()
        while queue:
            current, cost = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            local_visited.add(current)
            min_cost += cost
            for neighbor, edge_cost in graph[current]:
                if neighbor not in local_visited:
                    queue.append((neighbor, edge_cost))
    return min_cost

def main():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    W = list(map(int, sys.stdin.readline().strip().split()))

    # Create a mapping from items to the boxes they are in
    box_to_items = defaultdict(list)
    for i, box in enumerate(A):
        box_to_items[box].append(i + 1)

    # Sort the items in each box by their weight (in ascending order)
    for box in box_to_items:
        box_to_items[box].sort(key=lambda item: W[item - 1])

    # Calculate the minimum cost to move items to their correct boxes
    min_cost = find_min_cost(N, box_to_items, W)
    print(min_cost)

if __name__ == "__main__":
    main()