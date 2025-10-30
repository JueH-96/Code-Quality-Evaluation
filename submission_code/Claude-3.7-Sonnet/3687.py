class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Build the undirected graph
        graph = defaultdict(list)
        for u, v, length in edges:
            graph[u].append((v, length))
            graph[v].append((u, length))
        
        # Convert to a rooted tree (directed from root to leaves)
        children = defaultdict(list)
        
        def build_tree(node, parent):
            for neighbor, edge_length in graph[node]:
                if neighbor != parent:
                    children[node].append((neighbor, edge_length))
                    build_tree(neighbor, node)
        
        build_tree(0, -1)
        
        # Initialize the result
        max_length = 0
        min_nodes = float('inf')
        
        # Find the longest special path starting from a given node
        def find_longest_path(node, values_seen):
            # Check if the current node's value has been seen before
            if nums[node] in values_seen:
                return (0, float('inf'))  # Invalid path marker
            
            # Add current node's value to seen values
            new_values = values_seen.copy()
            new_values.add(nums[node])
            
            # Path with just the current node
            best_path = (0, 1)  # (length, nodes)
            
            # Try extending path to each child
            for child, edge_length in children[node]:
                child_path = find_longest_path(child, new_values)
                if child_path[1] != float('inf'):  # If child path is valid
                    new_path = (child_path[0] + edge_length, child_path[1] + 1)
                    if new_path[0] > best_path[0]:
                        best_path = new_path
                    elif new_path[0] == best_path[0]:
                        best_path = (best_path[0], min(best_path[1], new_path[1]))
            
            return best_path
        
        # Try starting from each node
        for node in range(n):
            path = find_longest_path(node, set())
            if path[1] != float('inf'):  # Only consider valid paths
                if path[0] > max_length:
                    max_length = path[0]
                    min_nodes = path[1]
                elif path[0] == max_length:
                    min_nodes = min(min_nodes, path[1])
        
        return [max_length, min_nodes]