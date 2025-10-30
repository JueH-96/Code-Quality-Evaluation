from collections import deque

def is_palindrome(s):
    return s == s[::-1]

def shortest_palindrome_path(adj_matrix, n, start, end):
    if start == end:
        return 0  # Path of length 0 is already a palindrome (empty string)
    
    queue = deque([(start, "", 0)])  # (vertex, concatenated_string, path_length)
    visited = set([(start, "")])
    
    while queue:
        vertex, string, path_length = queue.popleft()
        
        for next_vertex in range(1, n+1):
            edge_label = adj_matrix[vertex-1][next_vertex-1]
            if edge_label != '-':  # There's an edge
                new_string = string + edge_label
                new_path_length = path_length + 1
                
                if next_vertex == end and is_palindrome(new_string):
                    return new_path_length
                
                if (next_vertex, new_string) not in visited:
                    visited.add((next_vertex, new_string))
                    queue.append((next_vertex, new_string, new_path_length))
    
    return -1  # No valid path found

def main():
    N = int(input().strip())
    adj_matrix = []
    for _ in range(N):
        row = input().strip()
        adj_matrix.append(row)
    
    results = []
    for i in range(1, N+1):
        row_results = []
        for j in range(1, N+1):
            row_results.append(shortest_palindrome_path(adj_matrix, N, i, j))
        results.append(row_results)
    
    for row in results:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    main()