import sys

def are_segments_equal(s1s2, t1t2):
    # Define the adjacency list for the pentagon
    pentagon = {
        'A': ['B', 'E'],
        'B': ['A', 'C'],
        'C': ['B', 'D'],
        'D': ['C', 'E'],
        'E': ['D', 'A']
    }

    # Function to check if two points are adjacent
    def are_adjacent(p1, p2):
        return p2 in pentagon[p1]

    # Check if both segments are adjacent
    s1, s2 = s1s2
    t1, t2 = t1t2

    s1s2_adjacent = are_adjacent(s1, s2)
    t1t2_adjacent = are_adjacent(t1, t2)

    # If both segments are adjacent or both are not adjacent, they are equal
    return s1s2_adjacent == t1t2_adjacent

def main():
    input = sys.stdin.read
    data = input().split()

    s1s2 = data[0]
    t1t2 = data[1]

    if are_segments_equal(s1s2, t1t2):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()