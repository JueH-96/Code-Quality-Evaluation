def main():
    import sys
    input_data = sys.stdin.read().split()
    if len(input_data) < 2:
        return
    
    # Get the two segments from input
    seg1 = input_data[0].strip()  # S_1S_2
    seg2 = input_data[1].strip()  # T_1T_2

    # Create a mapping for the vertices of the pentagon
    # We assume the vertices are in order A, B, C, D, E.
    vertex_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    # Function to compute the chord "step" between two vertices in a regular pentagon.
    # The distance is determined by the minimum of (|i-j|) and (5 - |i-j|).
    def chord_distance(pair):
        first = vertex_map[pair[0]]
        second = vertex_map[pair[1]]
        diff = abs(first - second)
        return min(diff, 5 - diff)
    
    # Compute the step distances for both given segments.
    if chord_distance(seg1) == chord_distance(seg2):
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()