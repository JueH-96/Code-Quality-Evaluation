import sys

def solve(N, M, edges):
    # If there are no edges, we can assign any value to the vertices
    if M == 0:
        return "Yes
" + " ".join(["1"] * N)
    
    # If there is only one edge, it's impossible to satisfy the condition
    if M == 1:
        return "No"
    
    # If there are more than N-1 edges, it's impossible to satisfy the condition
    if M > N - 1:
        return "No"
    
    # Create a list to store the degree of each vertex
    degree = [0] * N
    for u, v in edges:
        degree[u - 1] += 1
        degree[v - 1] += 1
    
    # If there is a vertex with an odd degree, it's impossible to satisfy the condition
    if any(d % 2 == 1 for d in degree):
        return "No"
    
    # If all vertices have an even degree, we can assign the same value to all vertices
    return "Yes
" + " ".join(["1"] * N)

def main():
    N, M = map(int, sys.stdin.readline().split())
    edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]
    result = solve(N, M, edges)
    print(result)

if __name__ == "__main__":
    main()