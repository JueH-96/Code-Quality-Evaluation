# YOUR CODE HERE
def count_reachable_pairs(N, a):
    # Create an array to store the reachability count for each vertex
    reachable_count = [0] * N
    
    # Create an array to store the visited status of each vertex
    visited = [False] * N
    
    # Function to perform DFS and count reachable vertices
    def dfs(v):
        if visited[v]:
            return reachable_count[v]
        visited[v] = True
        next_vertex = a[v] - 1
        reachable_count[v] = 1 + dfs(next_vertex)
        return reachable_count[v]
    
    # Calculate reachable counts for all vertices
    for i in range(N):
        if not visited[i]:
            dfs(i)
    
    # Sum up all reachable counts
    total_reachable_pairs = sum(reachable_count)
    
    print(total_reachable_pairs)

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
a = list(map(int, data[1:]))

count_reachable_pairs(N, a)