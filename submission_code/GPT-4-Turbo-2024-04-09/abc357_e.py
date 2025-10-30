def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    a = list(map(int, data[1:]))
    
    # Adjusting for 1-based index
    a = [x - 1 for x in a]
    
    visited = [False] * N
    component_size = [0] * N
    component_id = [-1] * N
    current_component = 0
    
    # To find all cycles and their reachable nodes
    for start in range(N):
        if not visited[start]:
            # Detect cycle using Tortoise and Hare algorithm
            slow = fast = start
            while True:
                slow = a[slow]  # Move slow by 1
                fast = a[a[fast]]  # Move fast by 2
                if slow == fast:
                    break
            
            # Find the start of the cycle
            cycle_start = start
            while cycle_start != slow:
                cycle_start = a[cycle_start]
                slow = a[slow]
            
            # Count the size of the cycle
            cycle_size = 0
            cycle_node = cycle_start
            while True:
                cycle_size += 1
                visited[cycle_node] = True
                component_id[cycle_node] = current_component
                cycle_node = a[cycle_node]
                if cycle_node == cycle_start:
                    break
            
            # Mark all nodes in the cycle with the cycle size
            cycle_node = cycle_start
            while True:
                component_size[component_id[cycle_node]] = cycle_size
                cycle_node = a[cycle_node]
                if cycle_node == cycle_start:
                    break
            
            current_component += 1
    
    # Calculate the total number of reachable pairs
    result = 0
    for size in component_size:
        if size > 0:
            result += size * size
    
    print(result)

if __name__ == "__main__":
    main()