def main():
    import sys
    input_data = sys.stdin.read().split()
    if len(input_data) < 2:
        return

    # Map each vertex letter to an index in 0..4 corresponding the vertices of pentagon
    vertex_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # Read the two segments
    S = input_data[0].strip()  # e.g. "AC"
    T = input_data[1].strip()  # e.g. "EC"
    
    # Get the indexes
    s1, s2 = vertex_to_index[S[0]], vertex_to_index[S[1]]
    t1, t2 = vertex_to_index[T[0]], vertex_to_index[T[1]]
    
    def chord_distance(i, j):
        diff = abs(i - j)
        return min(diff, 5 - diff)
    
    # Compare the chord distances
    if chord_distance(s1, s2) == chord_distance(t1, t2):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()