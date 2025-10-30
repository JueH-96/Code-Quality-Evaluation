import sys

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    N, M = map(int, lines[0].split())
    friendships = []
    for line in lines[1:]:
        A, B = map(int, line.split())
        friendships.append((A, B))
    return N, M, friendships

def count_new_friendships(N, M, friendships):
    """Count the maximum number of new friendships."""
    # Create an adjacency list to represent the graph
    graph = [[] for _ in range(N + 1)]
    for A, B in friendships:
        graph[A].append(B)
        graph[B].append(A)

    # Initialize a set to keep track of new friendships
    new_friendships = set()

    # Iterate over all possible triplets (X, Y, Z)
    for X in range(1, N + 1):
        for Y in graph[X]:
            for Z in graph[Y]:
                # Check if X and Z are not friends
                if Z not in graph[X]:
                    # Add the new friendship to the set
                    new_friendships.add(tuple(sorted((X, Z))))

    return len(new_friendships)

def main():
    """Main function."""
    N, M, friendships = read_input()
    new_friendships = count_new_friendships(N, M, friendships)
    print(new_friendships)

if __name__ == "__main__":
    main()