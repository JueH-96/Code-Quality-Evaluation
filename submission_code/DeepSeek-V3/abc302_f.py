from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    sets = []
    for _ in range(N):
        A = int(data[idx])
        idx += 1
        S = list(map(int, data[idx:idx+A]))
        idx += A
        sets.append(S)
    
    # Create a graph where nodes are sets, and edges connect sets with common elements
    # To find the minimum number of operations, we need to find the shortest path from any set containing 1 to any set containing M
    
    # First, map each element to the sets that contain it
    element_to_sets = defaultdict(list)
    for i, s in enumerate(sets):
        for elem in s:
            element_to_sets[elem].append(i)
    
    # Now, we need to find the shortest path from any set containing 1 to any set containing M
    # We can perform a BFS starting from all sets containing 1, and see if we can reach any set containing M
    
    # Find all sets containing 1
    start_sets = set(element_to_sets[1])
    # Find all sets containing M
    end_sets = set(element_to_sets[M])
    
    if not start_sets or not end_sets:
        print(-1)
        return
    
    # If any set contains both 1 and M, no operations are needed
    for s in start_sets:
        if s in end_sets:
            print(0)
            return
    
    # Otherwise, perform BFS
    # We need to find the shortest path from any start_set to any end_set
    # We can model this as a graph where nodes are sets, and edges connect sets that share at least one common element
    
    # To build the graph, we need to find for each set, all other sets that share at least one element with it
    # This is computationally expensive, so we need a smarter approach
    
    # Instead of building the entire graph, we can perform BFS and for each set, find all sets that share elements with it on the fly
    
    # Initialize BFS
    visited = {}
    queue = deque()
    for s in start_sets:
        visited[s] = 0
        queue.append(s)
    
    while queue:
        current = queue.popleft()
        current_set = sets[current]
        # Find all elements in the current set
        elements = current_set
        # For each element, find all sets that contain it
        for elem in elements:
            for neighbor in element_to_sets[elem]:
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
                    if neighbor in end_sets:
                        print(visited[neighbor])
                        return
    
    # If we exhaust the queue and haven't found any end_set, it's impossible
    print(-1)

if __name__ == "__main__":
    main()