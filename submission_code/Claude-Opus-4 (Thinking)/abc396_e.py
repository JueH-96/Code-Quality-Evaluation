from collections import defaultdict, deque

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    graph = defaultdict(list)
    for _ in range(M):
        x, y, z = map(int, input().split())
        graph[x].append((y, z))
        graph[y].append((x, z))
    
    # Find connected components
    visited = [False] * (N + 1)
    components = []
    
    for i in range(1, N + 1):
        if not visited[i]:
            component = []
            queue = deque([i])
            visited[i] = True
            
            while queue:
                node = queue.popleft()
                component.append(node)
                
                for neighbor, _ in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            components.append(component)
    
    # Result array
    result = [0] * (N + 1)
    
    # Process each component
    for component in components:
        n = len(component)
        
        # For single isolated nodes, value is 0
        if n == 1 and component[0] not in graph:
            continue
        
        # For each bit position, find the optimal assignment
        component_result = [0] * n
        node_to_idx = {node: i for i, node in enumerate(component)}
        
        # Process each bit independently
        for bit in range(30):  # 30 bits is enough for values up to 10^9
            # Try both values for the first node's bit
            best_bit_count = float('inf')
            best_bit_assignment = None
            
            for start_bit in [0, 1]:
                bit_assignment = [-1] * n
                bit_assignment[0] = start_bit
                valid = True
                
                # BFS to assign bit values
                queue = deque([0])
                
                while queue and valid:
                    idx = queue.popleft()
                    node = component[idx]
                    
                    for neighbor, xor_value in graph[node]:
                        if neighbor in node_to_idx:
                            neighbor_idx = node_to_idx[neighbor]
                            xor_bit = (xor_value >> bit) & 1
                            expected_bit = bit_assignment[idx] ^ xor_bit
                            
                            if bit_assignment[neighbor_idx] == -1:
                                bit_assignment[neighbor_idx] = expected_bit
                                queue.append(neighbor_idx)
                            elif bit_assignment[neighbor_idx] != expected_bit:
                                valid = False
                                break
                
                if valid:
                    bit_count = sum(bit_assignment)
                    if bit_count < best_bit_count:
                        best_bit_count = bit_count
                        best_bit_assignment = bit_assignment
            
            if best_bit_assignment is None:
                print(-1)
                return
            
            # Update component result with this bit
            for i in range(n):
                if best_bit_assignment[i] == 1:
                    component_result[i] |= (1 << bit)
        
        # Assign values to result
        for i, node in enumerate(component):
            result[node] = component_result[i]
    
    print(' '.join(map(str, result[1:])))

solve()