# YOUR CODE HERE
import sys

def solve():
    """
    Reads two vertex pairs, determines if their connecting segments in a regular
    pentagon have equal length, and prints the result.
    """
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()

    def get_vertex_dist(p1, p2):
        """
        Calculates the shortest distance between two vertices along the perimeter
        of a regular pentagon.
        """
        # The length of a segment depends on the shortest number of sides
        # between its vertices. We can use the ordinal values of the
        # characters to represent the vertices numerically.
        diff = abs(ord(p1) - ord(p2))
        
        # The shortest path is the minimum of going one way or the other
        # around the pentagon (5 vertices). This will be 1 for adjacent
        # vertices (sides) and 2 for non-adjacent ones (diagonals).
        return min(diff, 5 - diff)

    # Calculate the characteristic vertex distance for each segment.
    dist_s = get_vertex_dist(s[0], s[1])
    dist_t = get_vertex_dist(t[0], t[1])

    # If the vertex distances are equal, the geometric lengths are equal.
    if dist_s == dist_t:
        print("Yes")
    else:
        print("No")

solve()