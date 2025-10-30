from sys import stdin

def is_good_pair(N, M, A, B):
    """
    Checks if a pair of sequences (A, B) is a good pair of sequences.

    Args:
    N (int): The maximum value of the elements in the sequences.
    M (int): The length of the sequences.
    A (list): The first sequence.
    B (list): The second sequence.

    Returns:
    bool: True if (A, B) is a good pair of sequences, False otherwise.
    """
    # Create a graph where each node represents an index in the sequence X
    # and there is an edge between two nodes if the corresponding elements in A and B are different
    graph = [[] for _ in range(N)]
    for i in range(M):
        graph[A[i] - 1].append(B[i] - 1)
        graph[B[i] - 1].append(A[i] - 1)

    # Initialize the color of each node to -1 (not colored)
    color = [-1] * N

    # Try to color the graph with two colors (0 and 1)
    for i in range(N):
        if color[i] == -1:
            if not dfs(graph, color, i, 0):
                return False

    return True

def dfs(graph, color, node, c):
    """
    Performs a depth-first search on the graph starting from the given node.

    Args:
    graph (list): The graph.
    color (list): The color of each node.
    node (int): The current node.
    c (int): The color to assign to the current node.

    Returns:
    bool: True if the graph can be colored, False otherwise.
    """
    if color[node] != -1:
        return color[node] == c
    color[node] = c
    for neighbor in graph[node]:
        if not dfs(graph, color, neighbor, 1 - c):
            return False
    return True

def main():
    N, M = map(int, stdin.readline().split())
    A = list(map(int, stdin.readline().split()))
    B = list(map(int, stdin.readline().split()))

    if is_good_pair(N, M, A, B):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()