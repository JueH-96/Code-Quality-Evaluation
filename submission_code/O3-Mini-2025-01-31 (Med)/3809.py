from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        # convert each property list to a set of unique integers
        sets = [set(prop) for prop in properties]
        
        # Union-Find helper functions
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX
        
        # Build the graph using unions: check pairs for common distinct integers >= k
        for i in range(n):
            for j in range(i+1, n):
                if len(sets[i].intersection(sets[j])) >= k:
                    union(i, j)
                    
        # count distinct roots, i.e., connected components
        components = len({find(i) for i in range(n)})
        return components

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    properties1 = [[1,2],[1,1],[3,4],[4,5],[5,6],[7,7]]
    print(sol.numberOfComponents(properties1, 1))  # Expected output: 3

    # Example 2
    properties2 = [[1,2,3],[2,3,4],[4,3,5]]
    print(sol.numberOfComponents(properties2, 2))  # Expected output: 1

    # Example 3
    properties3 = [[1,1],[1,1]]
    print(sol.numberOfComponents(properties3, 2))  # Expected output: 2