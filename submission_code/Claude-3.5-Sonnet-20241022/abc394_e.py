from collections import deque
import sys

def read_input():
    N = int(input())
    graph = []
    for _ in range(N):
        graph.append(list(input().strip()))
    return N, graph

def is_palindrome(s):
    return s == s[::-1]

def find_shortest_palindrome_path(N, graph, start, end):
    if start == end:
        return 0
    
    # BFS queue stores (current_vertex, path_length, path_string)
    queue = deque()
    # visited[vertex][path_string] stores minimum length to reach vertex with path_string
    visited = {}
    
    # Initialize with edges from start vertex
    for next_vertex in range(N):
        if graph[start][next_vertex] != '-':
            queue.append((next_vertex, 1, graph[start][next_vertex]))
    
    while queue:
        curr_vertex, length, path = queue.popleft()
        
        # If we reached the end vertex and path is palindrome
        if curr_vertex == end and is_palindrome(path):
            return length
            
        # If path is too long, skip
        if length >= 2*N:
            continue
            
        # Try all possible next edges
        for next_vertex in range(N):
            if graph[curr_vertex][next_vertex] != '-':
                new_path = path + graph[curr_vertex][next_vertex]
                new_length = length + 1
                
                # Check if this state was visited with a shorter path
                state = (next_vertex, new_path)
                if state not in visited or visited[state] > new_length:
                    visited[state] = new_length
                    queue.append((next_vertex, new_length, new_path))
    
    return -1

def main():
    N, graph = read_input()
    
    # Process each pair of vertices
    for i in range(N):
        result = []
        for j in range(N):
            shortest_length = find_shortest_palindrome_path(N, graph, i, j)
            result.append(str(shortest_length))
        print(" ".join(result))

if __name__ == "__main__":
    main()