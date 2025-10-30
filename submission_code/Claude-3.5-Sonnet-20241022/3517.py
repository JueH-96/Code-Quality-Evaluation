class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency matrix with infinity
        dist = [[float('inf')] * n for _ in range(n)]
        
        # Initialize direct paths
        for i in range(n):
            dist[i][i] = 0
            if i < n-1:
                dist[i][i+1] = 1
                
        # Function to run Floyd-Warshall algorithm
        def floyd_warshall():
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        # Run Floyd-Warshall first time for initial roads
        floyd_warshall()
        
        answer = []
        # Process each query
        for u, v in queries:
            # Add new road
            dist[u][v] = 1
            
            # Run Floyd-Warshall after each new road
            floyd_warshall()
            
            # Get shortest distance from 0 to n-1
            answer.append(dist[0][n-1])
            
        return answer