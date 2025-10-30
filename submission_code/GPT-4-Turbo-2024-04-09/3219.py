class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        from collections import defaultdict
        import heapq
        
        # Create a graph where each node is connected to other nodes with which it can swap
        graph = defaultdict(list)
        n = len(nums)
        
        # Sort nums with their indices
        sorted_nums = sorted((num, i) for i, num in enumerate(nums))
        
        # Use a min-heap to keep track of the next smallest element we can place in each position
        min_heap = []
        
        # Result array
        result = [0] * n
        
        # Visited array to keep track of placed elements
        visited = [False] * n
        
        # Build the graph based on the limit condition
        for i in range(n):
            num, idx = sorted_nums[i]
            # Connect this index with all indices within the limit range
            j = i
            while j < n and sorted_nums[j][0] - num <= limit:
                graph[idx].append(sorted_nums[j][1])
                j += 1
            j = i - 1
            while j >= 0 and num - sorted_nums[j][0] <= limit:
                graph[idx].append(sorted_nums[j][1])
                j -= 1
        
        # Function to perform DFS and collect all connected components that can be swapped
        def dfs(node, component):
            stack = [node]
            while stack:
                u = stack.pop()
                if visited[u]:
                    continue
                visited[u] = True
                component.append(u)
                for v in graph[u]:
                    if not visited[v]:
                        stack.append(v)
        
        # Go through each index, if not visited, perform DFS to find all swappable indices
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                # Extract the values from these indices
                values = [nums[x] for x in component]
                # Sort the values to place them in lexicographical order
                values.sort()
                # Sort the indices to place values in the correct positions
                component.sort()
                # Place sorted values into the result at the sorted indices
                for idx, value in zip(component, values):
                    result[idx] = value
        
        return result