import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:]))

    # Count the number of -1 in B
    q = B.count(-1)

    # Initialize the graph with N vertices and no edges
    graph = [[] for _ in range(N)]

    # Build the graph based on the given sequence B
    for i in range(N):
        for j in range(i + 1, N):
            if B[i] != -1 and B[j] != -1 and B[i] <= B[j]:
                graph[i].append(j)
                graph[j].append(i)

    # Function to find the number of connected components in the graph
    def count_connected_components(graph):
        visited = [False] * N
        components = 0

        def dfs(v):
            stack = [v]
            while stack:
                node = stack.pop()
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

        for i in range(N):
            if not visited[i]:
                components += 1
                visited[i] = True
                dfs(i)

        return components

    # Calculate the sum of f(B') over all possible B'
    total_sum = 0
    for mask in range(1 << q):
        # Generate a sequence B' based on the current mask
        B_prime = B.copy()
        idx = 0
        for i in range(N):
            if B[i] == -1:
                B_prime[i] = (mask >> idx) & 1
                idx += 1

        # Build the graph for the current B'
        graph_prime = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if B_prime[i] <= B_prime[j]:
                    graph_prime[i].append(j)
                    graph_prime[j].append(i)

        # Count the number of connected components in the current graph
        total_sum += count_connected_components(graph_prime)
        total_sum %= MOD

    print(total_sum)

if __name__ == "__main__":
    main()