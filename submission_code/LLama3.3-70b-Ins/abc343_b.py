import sys

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    return lines

def parse_input(lines):
    """Parse input lines into adjacency matrix and number of vertices."""
    N = int(lines[0].strip())
    A = []
    for i in range(1, N + 1):
        row = list(map(int, lines[i].strip().split()))
        A.append(row)
    return N, A

def find_connected_vertices(N, A):
    """Find directly connected vertices for each vertex."""
    connected_vertices = []
    for i in range(N):
        connected = [j + 1 for j, x in enumerate(A[i]) if x == 1]
        connected_vertices.append(connected)
    return connected_vertices

def print_connected_vertices(connected_vertices):
    """Print connected vertices for each vertex."""
    for connected in connected_vertices:
        print(' '.join(map(str, connected)))

def main():
    lines = read_input()
    N, A = parse_input(lines)
    connected_vertices = find_connected_vertices(N, A)
    print_connected_vertices(connected_vertices)

if __name__ == "__main__":
    main()