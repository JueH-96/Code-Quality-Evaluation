def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    reachable = [0] * (n + 1)
    visited = set()
    
    for u in range(1, n + 1):
        if u not in visited:
            path = []
            index_map = {}
            current = u
            while True:
                if current in visited:
                    # Process the path
                    for i in reversed(range(len(path))):
                        if i < len(path) - 1:
                            next_node = path[i + 1]
                        else:
                            next_node = current
                        node = path[i]
                        reachable[node] = 1 + reachable[next_node]
                    # Add all nodes in path to visited
                    for node in path:
                        visited.add(node)
                    break
                if current in index_map:
                    # Found a cycle
                    cycle_start = index_map[current]
                    cycle_size = len(path) - cycle_start
                    # Assign cycle nodes
                    for i in range(cycle_start, len(path)):
                        node = path[i]
                        reachable[node] = cycle_size
                    # Assign nodes before cycle
                    for i in range(cycle_start):
                        node = path[i]
                        reachable[node] = (cycle_start - i) + cycle_size
                    # Mark all nodes in path as visited
                    for node in path:
                        visited.add(node)
                    break
                # Add current to path and index_map
                index_map[current] = len(path)
                path.append(current)
                current = a[current - 1]  # a is 0-based list for input
    
    total = sum(reachable[1:n+1])
    print(total)

if __name__ == "__main__":
    main()